"""The Door's worlds in MuJoCo Warp on the GPU: the GPU physics that ``ndm_gym.wmbench.speed_door`` times beside the world
model and the native collector.

What differs from the native collection (ParallelNativeDoorCollector): MuJoCo Warp's float32 physics, and no no-slip solver
(MuJoCo Warp has none: the model's noslip_iterations are set to 0 here, and multi-contact and native CCD are disabled, which
MuJoCo Warp refuses with the model's contact margins; training and evaluation keep the native model with no-slip). The rest
follows the native collector: the control is mid + clip(a, -1, 1) half, five substeps per control step (one CUDA graph), a
world is done after the reference's last control and then restarts at its first state with a zero warm start (with
--door-latch-offset its door and latch raised by U(0, d) each, drawn on the device: another random stream), the phases
staggered from a replay of the reference's controls (here on the CPU with the same no-slip-free model). Observations,
actions and transitions stay on the device; ``step`` takes the actions as a tensor.
"""

import copy

import numpy as np
import torch

from .adroit_door import DoorPhysics

NCONMAX, NJMAX = 48, 256  # per world; the native replay with offsets and action noise peaks at 20 contacts, 120 rows
# The edges of an EPA horizon (MuJoCo Warp's MJ_MAX_EPAHORIZON is 24): with native CCD off, the convex pairs go through
# GJK/EPA, and a call can need more than 24 (it then stops early and raises the EPA_HORIZON overflow bit). A larger buffer
# changes nothing for a call that fits in 24.
EPA_HORIZON = 128


class WarpDoorCollector:
    on_device = True  # the actions stay on the device
    def __init__(self, model, reference, worlds, device='cuda:0', door_latch_offset=0., seed=0, epa_horizon=EPA_HORIZON):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        if type(worlds) is not int or worlds < 1:
            raise ValueError('worlds must be a positive integer')
        self.mjm = copy.copy(model)
        self.mjm.opt.noslip_iterations = 0
        # MuJoCo Warp refuses multi-contact and native CCD with the model's nonzero margins: both disabled
        self.mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD) | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.sim = DoorPhysics(self.mjm, reference)  # the native checks of layout, timestep and scene; the reset state
        self.reference, self.frame_skip, self.mjw, self.wp = reference, self.sim.frame_skip, mjw, wp
        self.tdev = torch.device(device)
        self.wdev = wp.get_device(f'cuda:{self.tdev.index or 0}')
        from mujoco_warp._src import collision_convex
        collision_convex.MJ_MAX_EPAHORIZON = int(epa_horizon)  # read where the EPA buffers are allocated (each collision call)
        with wp.ScopedDevice(self.wdev):
            self.m = mjw.put_model(self.mjm)
            self.d = mjw.put_data(self.mjm, self.sim.data, nworld=worlds, nconmax=NCONMAX, njmax=NJMAX)
        self.view = {k: wp.to_torch(getattr(self.d, k)) for k in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'time')}
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.tdev)  # noqa: E731
        self.mid, self.half = f(self.sim.mid), f(self.sim.half)
        self.q0, self.v0 = f(reference.qpos[0]), f(reference.qvel[0])
        self.last = reference.frames - 1
        self.frames = torch.zeros(worlds, dtype=torch.long, device=self.tdev)
        self.previous_actions = torch.zeros((worlds, 28), device=self.tdev)
        self.offset = float(door_latch_offset)
        self.generator = torch.Generator(device=self.tdev).manual_seed(seed * 1000003 + 20261116)
        self.transitions, self.worlds = 0, worlds
        self.graph = None
        self.stream = wp.stream_to_torch(wp.get_stream(self.wdev))
        self.bad = torch.zeros((), dtype=torch.bool, device=self.tdev)

    def _substeps(self):
        for _ in range(self.frame_skip):
            self.mjw.step(self.m, self.d)

    def _capture(self):
        """The first step: the kernels loaded by one eager control step, the state put back, the step captured (not run)."""
        wp = self.wp
        torch.cuda.synchronize(self.tdev)
        with wp.ScopedDevice(self.wdev):
            self._substeps()
            wp.synchronize_device()
            self._load(self._saved)
            torch.cuda.synchronize(self.tdev)
            with wp.ScopedCapture() as capture:
                self._substeps()
            self.graph = capture.graph

    def check(self):
        """After the steps: a nonfinite state, a full contact or constraint buffer, or any of MuJoCo Warp's
        overflow bits other than its solver's iteration limits (which the native solver also stops at) raises, naming them."""
        from mujoco_warp._src.types import OverflowType
        if bool(self.bad):
            raise FloatingPointError('Nonfinite MuJoCo Warp Door transition')
        d = self.d
        bits = 0
        for b in self.wp.to_torch(d.overflow).unique().tolist():
            bits |= int(b)
        bits &= ~int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
        if int(self.wp.to_torch(d.nacon).max()) >= d.naconmax or int(self.wp.to_torch(d.nefc).max()) >= d.njmax or bits:
            raise RuntimeError(f'MuJoCo Warp Door overflow: contacts {int(self.wp.to_torch(d.nacon).max())}/{d.naconmax}, rows '
                               f'{int(self.wp.to_torch(d.nefc).max())}/{d.njmax}, bits {OverflowType(bits)!r}')

    def observe(self):
        return dict(qpos=self.view['qpos'].clone(), qvel=self.view['qvel'].clone(), frames=self.frames.clone())

    def stagger_from_physical_replay(self, seed=0):
        """The native collector's phases: worlds at reference frames shuffled with numpy's generator from ``seed``, each in the
        state (and warm start) a replay of the reference's controls from frame 0 reaches there, on the CPU with this model."""
        if self.transitions:
            raise RuntimeError('Staggering requires a new collector')
        sim = DoorPhysics(self.mjm, self.reference)
        q, v, w = [], [], []
        for action in self.reference.actions:
            q.append(sim.data.qpos.copy())
            v.append(sim.data.qvel.copy())
            w.append(sim.data.qacc_warmstart.copy())
            sim.step(action)
        phases = np.arange(self.worlds) % len(self.reference.actions)
        np.random.default_rng(seed).shuffle(phases)
        f = lambda x: torch.as_tensor(np.asarray(x)[phases], dtype=torch.float32, device=self.tdev)  # noqa: E731
        previous = np.where((phases > 0)[:, None], self.reference.actions[(phases - 1).clip(0)].clip(-1, 1), 0.)
        self._load(dict(qpos=f(q), qvel=f(v), qacc_warmstart=f(w), frames=torch.as_tensor(phases, device=self.tdev),
                        previous_actions=torch.as_tensor(previous, dtype=torch.float32, device=self.tdev)))

    def _load(self, state):
        for key in ('qpos', 'qvel', 'qacc_warmstart'):
            self.view[key].copy_(state[key])
        self.frames.copy_(state['frames'])
        self.previous_actions.copy_(state['previous_actions'])
        self._saved = {k: v.clone() for k, v in state.items()}

    def state_dict(self):
        return dict(qpos=self.view['qpos'].cpu(), qvel=self.view['qvel'].cpu(), qacc_warmstart=self.view['qacc_warmstart'].cpu(),
                    frames=self.frames.cpu(), previous_actions=self.previous_actions.cpu(), transitions=self.transitions,
                    generator=self.generator.get_state())

    def load_state_dict(self, state):
        self._load({k: state[k].to(self.tdev) for k in ('qpos', 'qvel', 'qacc_warmstart', 'frames', 'previous_actions')})
        self.transitions = state['transitions']
        self.generator.set_state(state['generator'])

    @torch.no_grad()
    def step(self, actions):
        """One control step of every world from ``actions`` (worlds, 28) on the device: the native batch's tensors, then the
        done worlds restarted. A nonfinite state is flagged on the device and raised by ``check``."""
        actions = actions.clamp(-1, 1).float()
        before = self.observe()
        previous = self.previous_actions.clone()
        self.view['ctrl'].copy_(self.mid + actions * self.half)
        if self.graph is None:
            self._saved = dict(before, qacc_warmstart=self.view['qacc_warmstart'].clone(), previous_actions=previous)
            self._capture()
        current = torch.cuda.current_stream(self.tdev)
        self.stream.wait_stream(current)
        with self.wp.ScopedDevice(self.wdev):
            self.wp.capture_launch(self.graph)
        current.wait_stream(self.stream)
        next_q, next_v = self.view['qpos'].clone(), self.view['qvel'].clone()
        self.bad |= ~(torch.isfinite(next_q).all() & torch.isfinite(next_v).all())
        self.frames += 1
        done = self.frames >= self.last
        start_q = self.q0.expand_as(next_q).clone()
        if self.offset:
            start_q[:, 28:30] += self.offset * torch.rand((len(done), 2), device=self.tdev, generator=self.generator)
        self.view['qpos'].copy_(torch.where(done[:, None], start_q, next_q))
        self.view['qvel'].copy_(torch.where(done[:, None], self.v0.expand_as(next_v), next_v))
        self.view['qacc_warmstart'].masked_fill_(done[:, None], 0.)
        self.frames.masked_fill_(done, 0)
        self.previous_actions.copy_(torch.where(done[:, None], 0., actions))
        self.transitions += self.worlds
        return dict(**before, actions=actions, next_qpos=next_q, next_qvel=next_v, done=done, previous_actions=previous)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

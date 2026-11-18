"""Batched rollouts of mbrl-lib's environments (``sim.mbrl_envs``) for its true-dynamics controller's optimizer, on MuJoCo Warp
or on the NDM (``worldmodels.mjcf``). ``values(state, actions)`` scores action sequences (B, H, A) from one start state as
control_env.py's pool scores them on CPU MuJoCo (``mbrl.util.env.EnvHandler.rollout_env``): each sequence's undiscounted sum of
the environment's rewards, up to and including the step at which it terminates or its time limit truncates the rollout.

MuJoCo Warp: B worlds of the environment's own MjModel. An environment step sets ctrl and runs its physics steps (the frame
skip; dm_control: n_sub_steps per control step and ``repeat`` control steps, each followed by the kinematics and the subtrees'
centres of mass and velocities, as dm_control's step leaves them), then rne_postconstraint where a reward reads the contact
forces, as gymnasium's step does. What a reward reads before a step is the data then: the derived arrays of the last physics
step, as gymnasium's environments hold them, and of the start state itself at the start, as mbrl-lib's set_state (mj_forward)
leaves them. The constraint solver's warm start is zeroed at each start. On a GPU each control step's physics is one captured
Warp graph and the rewards run on its stream; on Warp's CPU device everything runs eagerly.

The NDM: one world-model step per control step (dm_control: ``repeat`` of them per environment step, with the same action, the
rewards summed as dm_control sums them), on qpos and qvel; the derived arrays a reward reads are
the forward kinematics of the world model's states (``mbrl_envs.Kinematics``; the pusher's named difference). Environments
whose rewards read contact forces are refused (``mbrl_envs.refusal``). On a GPU each control step (world model, kinematics,
reward, termination, the running sums) is one torch CUDA graph on static buffers, with the world model, kinematics, reward and
termination compiled as one function (``compile``); on the CPU it runs eagerly (``compile`` needs a GPU)."""

import contextlib
import types

import torch

from .mbrl_envs import Fields, Kinematics, MbrlEnv, Scorer, refusal


def _start(state: dict, B: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    q = torch.as_tensor(state["qpos"], dtype=torch.float32, device=device).expand(B, -1)
    v = torch.as_tensor(state["qvel"], dtype=torch.float32, device=device).expand(B, -1)
    goal = torch.as_tensor(state["goal"], dtype=torch.float32, device=device)[None] if "goal" in state else None
    return q, v, goal


def _derived(spec: MbrlEnv, when: str) -> list[str]:
    """The derived arrays (beyond qpos and qvel) the reward and termination read ``when`` (pre or post)."""
    return sorted(k.split(".")[1] for k in spec.reads if k.startswith(when + ".") and k not in (f"{when}.qpos", f"{when}.qvel"))


class WarpRollouts:
    """``values(state, actions)`` on ``nworld`` MuJoCo Warp worlds of ``mjm`` on Warp's ``device`` (\"cpu\" or \"cuda:N\")."""

    def __init__(self, spec: MbrlEnv, mjm, substeps: int, nworld: int, device: str = "cuda:0"):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp

        why = refusal(spec, "warp")
        if why:
            raise ValueError(f"WarpRollouts({spec.name}): refused, {why}")
        self.spec, self.substeps, self.nworld, self.mjw, self.wp = spec, substeps, nworld, mjw, wp
        self.device = wp.get_device(device)
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(mjm)
            self.d = mjw.put_data(mjm, mjd, nworld=nworld)
        self.tdev = torch.device("cuda", self.device.ordinal) if self.device.is_cuda else torch.device("cpu")
        self.scorer = Scorer(spec, mjm, substeps, self.tdev)
        d = self.d
        self.v = {k: wp.to_torch(getattr(d, k)) for k in ("qpos", "qvel", "ctrl", "xpos", "xmat", "xipos", "site_xpos", "subtree_linvel",
                                                         "cfrc_ext", "qacc_warmstart")}  # fmt: skip
        if spec.dm_control:
            self.stages = ("kinematics", "com_pos", "com_vel", "subtree_vel")
        else:
            self.stages = ("rne_postconstraint",) if "post.cfrc_ext" in spec.reads else ()
        self.pre_fields = _derived(spec, "pre")
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedDevice(self.device):
                self._control_step()  # the kernels load outside the capture
                wp.synchronize_device()
                with wp.ScopedCapture() as cap:
                    self._control_step()
            self.graph = cap.graph

    def _control_step(self):
        for _ in range(self.substeps):
            self.mjw.step(self.m, self.d)
        for s in self.stages:
            getattr(self.mjw, s)(self.m, self.d)

    @torch.no_grad()
    def values(self, state: dict, actions: torch.Tensor) -> torch.Tensor:
        """(B,) float32 on the CPU: the returns of ``actions`` (B, H, A) from ``state`` (sim.mbrl_envs.state)."""
        wp, spec, B = self.wp, self.spec, actions.shape[0]
        if B != self.nworld:
            raise ValueError(f"WarpRollouts: {B} sequences for {self.nworld} worlds")
        stream = wp.get_stream(self.device) if self.device.is_cuda else None
        ctx = torch.cuda.stream(wp.stream_to_torch(stream)) if stream is not None else contextlib.nullcontext()
        with wp.ScopedDevice(self.device), ctx:
            q, v, goal = _start(state, B, self.tdev)
            self.v["qpos"].copy_(q)
            self.v["qvel"].copy_(v)
            self.v["qacc_warmstart"].zero_()
            if self.pre_fields:  # the start state's kinematics, as set_state's mj_forward leaves them
                self.mjw.kinematics(self.m, self.d)
            acts = actions.to(self.tdev, torch.float32)
            total = torch.zeros(B, dtype=torch.float32, device=self.tdev)
            alive = torch.ones(B, dtype=torch.bool, device=self.tdev)
            post = Fields(**{k: self.v[k] for k in ("qpos", "qvel", "xpos", "xmat", "site_xpos", "subtree_linvel", "cfrc_ext")})
            for t in range(acts.shape[1]):
                u = acts[:, t]
                self.v["ctrl"].copy_(u)
                pre = Fields(**{k: self.v[k].clone() for k in ["qpos", "qvel"] + self.pre_fields})
                if spec.dm_control:
                    r = torch.zeros(B, dtype=torch.float32, device=self.tdev)
                    for k in range(spec.repeat):
                        self._replay()
                        if t * spec.repeat + k < state["control_left"]:  # dm_control's time limit ends the step's loop
                            r = r + self.scorer.reward(None, post, u)
                    end = (t + 1) * spec.repeat >= state["control_left"]
                    term = self.scorer.terminated(post) | end
                else:
                    self._replay()
                    r, term = self.scorer.reward(pre, post, u, goal), self.scorer.terminated(post)
                total += torch.where(alive, r, torch.zeros_like(r))
                alive &= ~term
                if t + 1 >= state["steps_left"]:  # the time limit truncates the rollout after this step
                    break
            return total.cpu()

    def _replay(self):
        if self.graph is not None:
            self.wp.capture_launch(self.graph)
        else:
            self._control_step()


class NdmRollouts:
    """``values(state, actions)`` with a world model (an MjcfWorldModel or its inference copy, on its device) in place of the
    physics: one world-model step per control step, which must be the model's step."""

    def __init__(self, spec: MbrlEnv, mjm, substeps: int, wm, nworld: int, compile: bool = False):  # noqa: A002
        why = refusal(spec, "ndm")
        if why:
            raise ValueError(f"NdmRollouts({spec.name}): refused, {why}")
        step_s = float(mjm.opt.timestep) * substeps
        if abs(wm.lay.dt - step_s) > 1e-9:
            raise ValueError(f"NdmRollouts({spec.name}): the world model steps {wm.lay.dt} s, the environment's control step is {step_s} s")
        self.spec, self.wm, self.nworld = spec, wm, nworld
        self.dev = wm.in_norm.mean.device
        if compile and self.dev.type != "cuda":
            raise ValueError("NdmRollouts: compile=True needs a GPU")
        self.scorer = Scorer(spec, mjm, substeps, self.dev)
        self.pre_names, self.post_names = _derived(spec, "pre"), _derived(spec, "post")
        self.kin = Kinematics(mjm, self.dev) if self.pre_names or self.post_names else None
        self.fn = torch.compile(self._score, mode="max-autotune-no-cudagraphs", dynamic=False) if compile else self._score
        nq, nv, nu = wm.lay.nq, wm.lay.nv, wm.lay.nu
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=self.dev)  # noqa: E731
        self.buf = types.SimpleNamespace(q=z(nworld, nq), v=z(nworld, nv), u=z(nworld, nu), goal=z(1, 3), total=z(nworld), alive=z(nworld))
        self.graph = self._capture() if self.dev.type == "cuda" else None

    def _fields(self, q, v, names) -> Fields:
        return Fields(qpos=q, qvel=v, **(self.kin(q, v, names) if names else {}))

    def _score(self, q, v, u, goal):
        """One control step: the next state, its reward and its termination."""
        q1, v1 = self.wm.step(q, v, u)
        post = self._fields(q1, v1, self.post_names)
        return q1, v1, self.scorer.reward(self._fields(q, v, self.pre_names), post, u, goal), self.scorer.terminated(post)

    def _body(self):
        b = self.buf
        q1, v1, r, term = self.fn(b.q, b.v, b.u, b.goal)
        b.total.add_(b.alive * r)
        b.alive.mul_((~term).float())
        b.q.copy_(q1)
        b.v.copy_(v1)

    def _capture(self):
        from ..worldmodels.mjcf import fused_compile

        side = torch.cuda.Stream(device=self.dev)
        side.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.no_grad(), torch.cuda.stream(side), fused_compile(self.wm):
            for _ in range(3):  # compilation and the allocator's warm-up, outside the capture
                self._body()
        torch.cuda.current_stream(self.dev).wait_stream(side)
        torch.cuda.synchronize(self.dev)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            self._body()
        return graph

    @torch.no_grad()
    def values(self, state: dict, actions: torch.Tensor) -> torch.Tensor:
        b, B, spec = self.buf, actions.shape[0], self.spec
        if B != self.nworld:
            raise ValueError(f"NdmRollouts: {B} sequences for {self.nworld} worlds")
        q, v, goal = _start(state, B, self.dev)
        b.q.copy_(q)
        b.v.copy_(v)
        if goal is not None:
            b.goal.copy_(goal)
        b.total.zero_()
        b.alive.fill_(1.0)
        acts = actions.to(self.dev, torch.float32)
        for t in range(min(acts.shape[1], state["steps_left"])):  # the time limit truncates the rollout after steps_left steps
            b.u.copy_(acts[:, t])
            inner = min(spec.repeat, state["control_left"] - t * spec.repeat) if spec.dm_control else 1  # dm_control: up to its limit
            for _ in range(inner):
                if self.graph is not None:
                    self.graph.replay()
                else:
                    self._body()
            if spec.dm_control and (t + 1) * spec.repeat >= state["control_left"]:  # dm_control's time limit ends the episode
                break
        return b.total.cpu()


def load_ndm_copy(path: str, mjm, device, dtype=torch.bfloat16):
    """A fit_mjcf --save file's world model of ``mjm`` (its planar root, step and network), as an inference copy in ``dtype``
    (None: fp32) on ``device``."""
    from ..wmbench.speed_mjcf import load_ndm
    from ..worldmodels.mjcf import MjcfLayout, inference_copy

    return inference_copy(load_ndm(MjcfLayout.from_model(mjm), types.SimpleNamespace(ckpt=path), device), dtype)

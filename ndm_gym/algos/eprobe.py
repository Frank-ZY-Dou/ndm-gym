# Ported from BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658, src/blinkmimic/train.py: --eprobe_*) under its
# MIT License:
#
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Evaluation-guided resets (``--eprobe_share`` and the other ``--eprobe_*`` options; BlinkMimic).

Every ``--eprobe_every`` iterations from ``--eprobe_start``, the deployed policy (the ``--pi_ema`` average, deterministic)
is swept over all clips under evaluation conditions: one probe world per ``--eprobe_tile`` frames of each clip, started a
random lead (one per sweep, in [``--eprobe_lead_min``, ``--eprobe_lead_max``]) frames earlier on the reference, run until the
tile's end. The probes' first failures (termination, or a non-finite action) add to decayed per-bin counts
(``--eprobe_decay`` per sweep). Each bin then weighs the counts of itself and the ``--eprobe_win`` bins after it in its clip,
i.e. the window before each failure. A share ``--eprobe_share`` of the training resets follows these weights
(``TrackingSim.set_focus``).

BlinkMimic sweeps in a second, evaluation-mode simulator of its training process. Isaac Lab 3 holds one SimulationContext
per process, so here the probe simulator runs in a worker process on the same GPU: ``python -m ndm_gym.algos.eprobe --worker
<dir> <args.json>``, a ``TrackingSim`` in evaluation mode, as ``ndm_gym.evaluate`` builds it. The trainer sends the policy
average at the sweep's start and applies the failures at the iteration where BlinkMimic does. With ``--eprobe_async``, that
is ``ceil(steps / --eprobe_async)`` iterations later (``steps`` = lead + tile; the first sweep three fewer, BlinkMimic's
graph warm-up), waiting if the worker is not done. Without it, the same iteration. ``--eprobe_graph`` replays one sweep step
as a CUDA graph and runs all ``steps`` (BlinkMimic's ``--eprobe_graph``); without it the sweep stops once every probe is done.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

from ..motion.library import MotionLib

READY, STOP = "ready", "stop"


def add_eprobe_args(g):
    """The ``--eprobe_*`` options (off by default), in BlinkMimic's argument group (algos.blinkmimic)."""
    g.add_argument("--eprobe_share", type=float, default=0.0,
                   help="any algorithm, with --pi_ema: this share of the resets goes to the --eprobe_win bins before the places where "
                   "the deployed policy (the average, deterministic, evaluation conditions) fails in a sweep over all clips "
                   "(algos/eprobe.py; 0: off)")  # fmt: skip
    g.add_argument("--eprobe_tile", type=int, default=300, help="with --eprobe_share: frames per probe world")
    g.add_argument("--eprobe_lead_min", type=int, default=200, help="with --eprobe_share: least frames a probe starts before its tile")
    g.add_argument("--eprobe_lead_max", type=int, default=450, help="with --eprobe_share: most frames a probe starts before its tile")
    g.add_argument("--eprobe_every", type=int, default=250, help="with --eprobe_share: iterations between sweeps")
    g.add_argument("--eprobe_start", type=int, default=1250, help="with --eprobe_share: the first sweep's iteration")
    g.add_argument("--eprobe_win", type=int, default=10, help="with --eprobe_share: bins (1 s each) before a failure that receive its weight")
    g.add_argument("--eprobe_decay", type=float, default=0.5, help="with --eprobe_share: per-sweep decay of the failure counts")
    g.add_argument("--eprobe_graph", action="store_true",
                   help="with --eprobe_share: one sweep step replayed as a CUDA graph, all lead + tile steps run (the same values)")  # fmt: skip
    g.add_argument("--eprobe_async", type=int, default=0,
                   help="with --eprobe_graph: a sweep's failures apply ceil(steps / this) iterations after its start, as BlinkMimic's "
                   "sweep on a side stream at this many steps per iteration (0: the same iteration)")  # fmt: skip


def check_eprobe_args(args: argparse.Namespace):
    """The ``--eprobe_*`` options' constraints; raises ValueError."""
    if not 0 <= args.eprobe_share < 1:
        raise ValueError(f"--eprobe_share must be in [0, 1), got {args.eprobe_share}")
    if args.eprobe_share == 0:
        if args.eprobe_graph or args.eprobe_async:
            raise ValueError("--eprobe_graph and --eprobe_async need --eprobe_share")
        return
    if args.pi_ema <= 0:
        raise ValueError("--eprobe_share sweeps the policy average: needs --pi_ema")
    if args.eprobe_tile < 1 or args.eprobe_every < 1 or args.eprobe_win < 0 or not 0 <= args.eprobe_lead_min <= args.eprobe_lead_max:
        raise ValueError("--eprobe_tile >= 1, --eprobe_every >= 1, --eprobe_win >= 0, 0 <= --eprobe_lead_min <= --eprobe_lead_max")
    if not 0 <= args.eprobe_decay <= 1:
        raise ValueError(f"--eprobe_decay must be in [0, 1], got {args.eprobe_decay}")
    if args.eprobe_async < 0 or (args.eprobe_async and not args.eprobe_graph):
        raise ValueError("--eprobe_async needs --eprobe_graph")
    if args.paired:
        raise ValueError("--eprobe_share: not implemented for the two-robot scene (--paired)")
    if args.eprobe_async and math.ceil((args.eprobe_lead_max + args.eprobe_tile) / args.eprobe_async) >= args.eprobe_every:
        raise ValueError("--eprobe_async too small: a sweep would not be applied before the next one starts")


def probe_tiles(motion: MotionLib, tile: int) -> torch.Tensor:
    """The probe worlds' tile starts: every ``tile`` frames of each clip, up to one before its last frame."""
    return torch.cat([torch.arange(int(motion.clip_start[c]), int(motion.clip_end[c]) - 1, tile) for c in range(len(motion.names))]).to(
        motion.clip_start.device
    )


def sweep_frames(motion: MotionLib, tiles: torch.Tensor, lead: int, tile: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(start, stop, cap) frames of a sweep with this lead: each probe starts ``lead`` frames before its tile (not before
    its clip), is done at the tile's end or the clip's last frame, and then keeps stepping with its frame capped at one
    before the clip's last (so that frame + 1 stays in the clip)."""
    cid = motion.clip_id[tiles]
    f0 = torch.maximum(tiles - lead, motion.clip_start[cid])
    stop = torch.minimum(tiles + tile, motion.clip_end[cid])
    cap = torch.minimum(stop, motion.clip_end[cid] - 1)
    return f0, stop, cap


def next_bins(motion: MotionLib) -> torch.Tensor:
    """Each bin's next bin in its clip; a clip's last bin maps to itself."""
    nxt = torch.arange(motion.num_bins, device=motion.bin_clip.device) + 1
    same = (nxt < motion.num_bins) & (motion.bin_clip[nxt.clamp(max=motion.num_bins - 1)] == motion.bin_clip)
    return torch.where(same, nxt, nxt - 1)


class FailureFocus:
    """BlinkMimic's ``EP_FAIL`` and the focus distribution made of it (its ``eprobe_finish``)."""

    def __init__(self, motion: MotionLib, win: int, decay: float):
        self.motion, self.win, self.decay = motion, win, decay
        self.counts = torch.zeros(motion.num_bins, device=motion.bin_clip.device)  # decayed counts of first failures per bin
        self.next = next_bins(motion)

    def update(self, fail_f: torch.Tensor) -> torch.Tensor | None:
        """One sweep's first-failure frames (-1: no failure): the counts decay and add one per failure in its frame's bin.
        Returns the focus distribution over bins (each bin: the counts of itself and the ``win`` bins after it in its
        clip, normalized), or None while there is no failure."""
        self.counts.mul_(self.decay)
        fb = self.motion.frame_bin[fail_f[fail_f >= 0].to(self.counts.device)]
        self.counts.index_add_(0, fb, torch.ones_like(fb, dtype=self.counts.dtype))
        if not self.counts.sum() > 0:
            return None
        w, q = self.counts.clone(), self.counts.clone()
        own = torch.arange(self.motion.num_bins, device=w.device)
        for _ in range(self.win):  # q[b] = sum of the counts over b .. b + win (same clip)
            w = w[self.next] * (self.next != own).float()
            q += w
        return q / q.sum()


def probe_action(pi, hist: torch.Tensor, goal: torch.Tensor, residual: bool, ref_u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """The deployed policy's setpoint in a probe world, as ndm_gym.evaluate computes it: no noise; a residual policy's
    output about the next reference pose."""
    a = pi(hist, goal)
    return a + ref_u[f + 1] if residual else a


class ProbeSweep:
    """The probe simulator's side (worker process): ``TrackingSim`` in evaluation mode with one world per tile."""

    def __init__(self, env, pi, motion: MotionLib, mdp, K: int, Hh: int, residual: bool, ref_u: torch.Tensor, tiles: torch.Tensor,
                 tile: int, graph: bool):  # fmt: skip
        self.env, self.pi, self.motion, self.mdp, self.K, self.Hh = env, pi, motion, mdp, K, Hh
        self.residual, self.ref_u, self.tiles, self.tile, self.graph = residual, ref_u, tiles, tile, graph
        P, dev = len(tiles), env.device
        self.ids = torch.arange(P, device=dev)
        self.fail_f = torch.full((P,), -1, dtype=torch.long, device=dev)
        self.stop = torch.zeros(P, dtype=torch.long, device=dev)
        self.cap = torch.zeros(P, dtype=torch.long, device=dev)
        self.hist = None
        self.cuda_graph = self.vel = None
        self.sweeps = 0

    def _step(self, S):
        """One sweep step from the state S just read: the policy, the simulator's step, the first failures, the probes done
        at their stop, the frames capped. In place (a CUDA graph replays it)."""
        env = self.env
        o = self.mdp.obs(S, env.a_prev, env.xi, env.f)
        self.hist.copy_(torch.cat((self.hist[:, 1:], o.unsqueeze(1)), 1))
        a = probe_action(self.pi, self.hist, self.motion.goal(env.f, self.K), self.residual, self.ref_u, env.f)
        alive = ~env.done_eval
        f_now = env.f.clone()
        out = env.step(a, S, inplace=True)
        self.fail_f.copy_(torch.where(alive & (out["term"] | ~torch.isfinite(a).all(-1)), f_now, self.fail_f))
        env.done_eval |= env.f >= self.stop
        env.f.copy_(torch.minimum(env.f, self.cap))

    @torch.no_grad()
    def run(self, lead: int) -> torch.Tensor:
        """One sweep with this lead: the probes' first-failure frames (-1: none), on the CPU."""
        from ..sim.rigid.isaaclab_mjwarp import after_replay, before_replay, capture_step, on_capture_stream
        from ..state import CONTROL_DT

        env = self.env
        f0, stop, cap = sweep_frames(self.motion, self.tiles, lead, self.tile)
        self.stop.copy_(stop), self.cap.copy_(cap), self.fail_f.fill_(-1)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(self.ids, f0)
        n = lead + self.tile
        S = env.state()
        o = self.mdp.obs(S, env.a_prev, env.xi, env.f)
        if self.hist is None:
            self.hist = o.unsqueeze(1).repeat(1, self.Hh, 1)
        else:
            self.hist.copy_(o.unsqueeze(1).expand(-1, self.Hh, -1))
        if not self.graph:  # eager: until every probe is done (the first step shifts o_0 in again: the history is unchanged)
            for _ in range(n):
                self._step(S)
                if bool(env.done_eval.all()):
                    break
                S = env.state()
        else:  # every step, as evaluate --graph (_loop_graph): the first sweep's steps 1-2 eagerly (2 on the capture stream),
            # step 3 captured; every later step a replay of it, on the fixed tensors the capture recorded
            vel = env.robot.data._root_link_vel_w.data  # the root velocity property's buffer, made current before each step
            if self.vel is None:
                self.vel = vel
            elif vel is not self.vel:
                raise RuntimeError("the root velocity buffer the probe step was captured on was replaced")

            def body():
                self._step(env.state(self.vel))

            for t in range(1, n + 1):
                env.robot.data.root_link_vel_w  # recomputed if stale, else kept, exactly as state() decides
                if self.cuda_graph is None and t == 1:
                    body()
                elif self.cuda_graph is None and t == 2:
                    on_capture_stream(env.sim, body)
                else:
                    before_replay(env.sim)
                    first = self.cuda_graph is None
                    if first:  # the capture runs the step's Python side once; the replay below does its GPU work
                        self.cuda_graph, _, self._record = capture_step(env.sim, body)
                    self.cuda_graph.replay()
                    if not first:
                        after_replay(env.sim, env.scene, CONTROL_DT, 1)
        self.sweeps += 1
        return self.fail_f.cpu()


class EvalProbe:
    """The trainer's side: the worker process, the sweeps' timing and the focus they set on ``env``."""

    def __init__(self, args: argparse.Namespace, motion: MotionLib, env, log_dir: str):
        self.args, self.motion, self.env = args, motion, env
        self.tiles = probe_tiles(motion, args.eprobe_tile)
        self.focus = FailureFocus(motion, args.eprobe_win, args.eprobe_decay)
        self.dir = tempfile.mkdtemp(prefix="eprobe_", dir=os.environ.get("TMPDIR"))
        self.log = open(os.path.join(log_dir, "eprobe_worker.log"), "a")
        cmd = [sys.executable, "-m", "ndm_gym.algos.eprobe", "--worker", self.dir, os.path.join(log_dir, "args.json")]
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        self.k = 0  # requests sent
        self.pending = None  # (k, start iteration, lead, apply iteration, t0)
        print(f"--eprobe: {len(self.tiles)} probe worlds (tiles of {args.eprobe_tile} frames) in a worker process ({self.dir})", flush=True)

    def _path(self, kind: str, k: int) -> str:
        return os.path.join(self.dir, f"{kind}_{k:05d}.pt")

    def _wait(self, path: str) -> dict:
        while not os.path.exists(path):
            if self.proc.poll() is not None:
                self.log.flush()
                raise RuntimeError(f"the probe worker exited ({self.proc.returncode}); see eprobe_worker.log in the run's directory")
            time.sleep(0.005)
        return torch.load(path, map_location="cpu")

    def tick(self, it: int, pi_avg_module):
        """Iteration ``it``, before its collection: a running sweep's failures applied when due, then a new sweep
        started on the sweep iterations (BlinkMimic's eprobe_async_tick and eprobe_burst, in its order)."""
        a = self.args
        if self.pending is not None and it >= self.pending[3]:
            self._apply(it)
        if it >= a.eprobe_start and it % a.eprobe_every == 0:
            if self.pending is not None:
                raise RuntimeError("--eprobe_async too small: a sweep is still pending")
            lead = int(torch.randint(a.eprobe_lead_min, a.eprobe_lead_max + 1, (1,)))
            steps = lead + a.eprobe_tile - (3 if a.eprobe_graph and self.k == 0 else 0)  # BlinkMimic: the graph warm-up's 3 steps
            sd = {k: v.detach().to("cpu", copy=True) for k, v in pi_avg_module.state_dict().items()}
            path = self._path("req", self.k)
            torch.save({"pi": sd, "lead": lead, "it": it}, path + ".tmp")
            os.replace(path + ".tmp", path)
            due = it + math.ceil(steps / a.eprobe_async) if a.eprobe_async else it
            self.pending = (self.k, it, lead, due, time.time())
            self.k += 1
            if not a.eprobe_async:
                self._apply(it)

    def _apply(self, it: int):
        k, it0, lead, _, t0 = self.pending
        res = self._wait(self._path("res", k))
        os.remove(self._path("res", k))
        self.pending = None
        fail_f = res["fail_f"]
        q = self.focus.update(fail_f)
        if q is not None:
            self.env.set_focus(q, self.args.eprobe_share)
        m = self.motion
        ff = fail_f[fail_f >= 0].to(m.clip_id.device)
        fl = sorted({(m.names[int(m.clip_id[f])].replace("_subject", "_s"), int(f - m.clip_start[m.clip_id[f]])) for f in ff.tolist()})
        print(f"--eprobe it {it0}: lead {lead} | {len(ff)}/{len(self.tiles)} failures {fl[:8]} | sweep {res['sweep_s']:.1f} s, "
              f"{time.time() - t0:.1f} s to apply | applied at it {it}", flush=True)  # fmt: skip

    def state_dict(self) -> dict:
        """For a checkpoint: the failure counts and the focus with its share (a sweep in flight is not kept, as BlinkMimic's)."""
        env = self.env
        share = float(env.focus_share_t) if getattr(env, "focus_static", False) else env.focus_share
        return {"eprobe_fail": self.focus.counts.cpu(), "eprobe_focus": None if env.focus_q is None else env.focus_q.cpu(),
                "eprobe_focus_share": share}  # fmt: skip

    def load_state_dict(self, ck: dict):
        if "eprobe_fail" not in ck:
            raise ValueError("--eprobe_share: the checkpoint has no probe state (eprobe_fail)")
        self.focus.counts.copy_(ck["eprobe_fail"].to(self.focus.counts.device))
        if ck["eprobe_focus"] is not None:
            self.env.set_focus(ck["eprobe_focus"].to(self.focus.counts.device), ck["eprobe_focus_share"])

    def close(self):
        """Stop the worker (its exit is waited for; a worker that does not stop is killed) and remove the request
        directory. A second call does nothing."""
        if self.log.closed:
            return
        if self.proc.poll() is None:
            open(os.path.join(self.dir, STOP), "w").close()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def worker(qdir: str, args_json: str):
    """The probe simulator: builds the motion library, the evaluation's MDP and an evaluation-mode TrackingSim with one
    world per tile from the run's arguments, then serves the trainer's sweep requests in order until told to stop."""
    from isaaclab.app import launch_simulation

    import ndm_gym.core.nets as nets
    from ndm_gym.algos import ALGOS
    from ndm_gym.algos.common import reference_setpoints
    from ndm_gym.motion.sources import lafan1, read_clip_list, select
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.robots import g1
    from ndm_gym.sim.rigid.isaaclab_mjwarp import TrackingSim, make_sim
    from ndm_gym.tasks.tracking import MDP, OBS_DIM

    parent = os.getppid()  # the trainer: a worker whose trainer is gone exits
    a = argparse.Namespace(**json.load(open(args_json)))
    if a.motions != "lafan1":
        raise ValueError(f"the probe worker builds LAFAN1 libraries only, not {a.motions}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    nets.AMP_DTYPE = torch.bfloat16 if a.amp == "bf16" else None
    dev = "cuda:0"
    kin = g1.kinematics(dev)
    clips = lafan1.load_dir(a.motion_dir)
    if a.clips:
        clips = select(clips, read_clip_list(a.clips))
    motion = MotionLib(clips, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(g1.default_joint_pos_list(), device=dev))  # the evaluation's (ndm_gym.evaluate)
    tiles = probe_tiles(motion, a.eprobe_tile)
    sim_cfg = make_sim(dev, a)
    with launch_simulation(sim_cfg, a):
        env = TrackingSim(sim_cfg, len(tiles), motion, mdp, eval_mode=True)
        pi = TrackingPolicy(OBS_DIM, 73 * a.goal_frames, hist=a.hist).to(dev)
        pi.eval()
        residual = ALGOS[a.algo].is_residual(vars(a))
        sweep = ProbeSweep(env, pi, motion, mdp, a.goal_frames, a.hist, residual, reference_setpoints(motion, mdp.default_q, env.alpha),
                           tiles, a.eprobe_tile, a.eprobe_graph)  # fmt: skip
        open(os.path.join(qdir, READY), "w").close()
        print(f"probe worker ready: {len(tiles)} worlds", flush=True)
        k = 0
        while True:
            req = os.path.join(qdir, f"req_{k:05d}.pt")
            while not os.path.exists(req):
                if os.path.exists(os.path.join(qdir, STOP)):
                    print("probe worker stopped", flush=True)
                    return
                if os.getppid() != parent:  # the trainer died without stopping the worker (e.g. killed)
                    print("probe worker: the trainer is gone, exiting", flush=True)
                    return
                time.sleep(0.005)
            r = torch.load(req, map_location=dev)
            os.remove(req)
            pi.load_state_dict(r["pi"])
            t0 = time.time()
            fail_f = sweep.run(r["lead"])
            res = os.path.join(qdir, f"res_{k:05d}.pt")
            torch.save({"fail_f": fail_f, "sweep_s": time.time() - t0}, res + ".tmp")
            os.replace(res + ".tmp", res)
            k += 1


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("usage: python -m ndm_gym.algos.eprobe --worker <request dir> <run's args.json>")
    worker(sys.argv[2], sys.argv[3])

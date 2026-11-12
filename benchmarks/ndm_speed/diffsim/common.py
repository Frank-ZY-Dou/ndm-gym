"""Shared pieces of the differentiable-simulator speed benchmarks (standard library + numpy + mujoco only, so every
simulator's environment can import it).

The fixed workload (identical in every bench_*.py):
  model      the mjlab Unitree G1 that NablaEngine bundles (models/mjlab_g1/g1_train.mjb, MuJoCo 3.17.0 binary: 29 PD
             position actuators, 34 primitive collision geoms, 502 explicit contact pairs, Newton, pyramidal cone,
             implicitfast, timestep 5 ms), sensors disabled (physics only)
  step       one environment step = one 20 ms control step = 4 substeps of 5 ms with the PD targets held
  precision  float32
  states     standing states: the model's keyframe (knees_bent if present, else home) with joint noise, the root
             lifted by 5 mm, small joint velocities; the same arrays for every simulator (seeded numpy generator)
  targets    the keyframe's joint angles plus a smooth random offset per world, joint and step (AR(1), sigma 0.05 rad,
             correlation 0.9 between control steps), clipped to the actuators' ctrlrange
  forward    (a) STEPS control steps (default 200) as SEGMENTS of SEGMENT_STEPS (default 4 x 50): every segment
             restarts from the standing states, so the robots stay upright (no balance controller is running);
             control steps per second = nworld * STEPS / time, median of REPEATS (default 3) repeats
  gradient   (b) forward + reverse-mode gradient of L = 1/N sum_worlds sum_{t=1..H} [0.5 |q_joints(t) - q_ref|^2 +
             0.5 W_V |qvel(t)|^2] over H = 32 control steps with respect to the H target vectors of every world;
             ms per control step = time / H, median of REPEATS repeats; peak GPU memory of the process
  settings   (c) solver iterations (cap), line-search iterations, tolerance, contact / constraint capacities, and
             whether any world overflowed them: one CSV line per measurement holds all of them
  loss       the column `loss` of a gradient line is L, with the same definition in every script: at equal nworld,
             horizon and seed the simulators should agree closely, a check that they computed the same physics

Each bench script prints one CSV header line and one CSV line per measurement on stdout (prefix "CSV,"), and appends
the same line to --csv FILE when given.
"""

import csv
import datetime
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("DIFFSIM") or Path(__file__).resolve().parents[1])
MODEL_DIR = ROOT / "models" / "mjlab_g1"
MODEL_FILE = MODEL_DIR / "g1_train.mjb"
META_FILE = MODEL_DIR / "g1_meta.json"
MUJOCO_VERSION = "3.17.0"   # the binary model's format
SUBSTEPS = 4
TIMESTEP = 0.005
NQ, NV, NU = 36, 35, 29
W_V = 0.01                  # weight of the velocity term of the rollout loss
# The Newton iteration cap of the --tuned modes: the value the model file stores (Menagerie's MJX options). On this
# workload MuJoCo C (float64, 32 worlds, check of 2026-11-10) gives the same rollout with a cap of 5 as with 10
# (loss per world 3.3027 against 3.3051 over 32 control steps, largest joint speed 3.15 rad/s in both; mean 2.6
# iterations per substep), while lower caps change it (3: 4.36, 2: 21.3) and a cap of 1 is unstable (speeds above
# 180 rad/s after two control steps, "Nan, Inf or huge value in QACC" within 0.2-0.6 s).
# The float32 engines use more iterations than float64 MuJoCo C, mostly in the first substeps after the cold start:
# in the CPU smoke test (2 worlds, 2 control steps) MuJoCo Warp had 1 of 2 worlds at a cap of 5 in some substep and a
# loss 0.11 % away from the cap-10 run. So a cap of 5 is a small change of the solve, not none: read the column
# iter_limit_worlds, and pass --iterations 10 to a --tuned run for the same cap as the authors' settings.
TUNED_ITERATIONS = 5
FALL_Z = 0.4                # a world counts as fallen when its root is below this height [m]
SIGMA_Q0, SIGMA_V0, LIFT = 0.03, 0.05, 0.005
SIGMA_U, RHO_U = 0.05, 0.9

COLUMNS = (
    "sim", "sim_version", "sim_commit", "mode", "measure", "nworld", "steps", "segment_steps", "horizon", "repeats",
    "substeps", "timestep", "precision", "solver", "cone", "integrator", "iterations", "ls_iterations", "tolerance",
    "capacity_contacts", "capacity_rows", "capacity_note", "overflow_worlds", "iter_limit_worlds", "solver_iter_mean",
    "contacts_max", "rows_max", "status", "env_steps_per_s", "ms_per_control_step", "repeat_values", "repeat_unit",
    "compile_s", "peak_mem_framework_mib", "peak_mem_process_mib", "gpu_mem_total_mib", "fallen_worlds", "finite",
    "loss", "grad_abs_mean", "knobs", "gpu", "driver", "python", "packages", "seed", "utc", "notes")


def parser(description):
    import argparse
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tuned", action="store_true", help="every speed knob of the header turned on")
    g.add_argument("--as_published", action="store_true", help="the simulator authors' own settings")
    ap.add_argument("--nworld", type=int, required=True, help="number of worlds (1024, 4096, 16384)")
    ap.add_argument("--measure", default="both", choices=["forward", "grad", "both"],
                    help="forward: (a); grad: (b); both: (a) then (b) in one process (the peak memory of (b) is "
                         "then an upper bound: run --measure grad alone for the exact number)")
    ap.add_argument("--steps", type=int, default=200, help="control steps per forward repeat")
    ap.add_argument("--segment_steps", type=int, default=50, help="control steps between restarts from the standing states")
    ap.add_argument("--horizon", type=int, default=32, help="control steps of the differentiated rollout")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=None, help="solver iteration cap (default: per mode, see header)")
    ap.add_argument("--ls_iterations", type=int, default=None, help="line-search iterations (default: per mode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="append the CSV lines to this file (header written when it is new)")
    ap.add_argument("--smoke_cpu", action="store_true",
                    help="development only: run on the CPU with tiny sizes to check the code path; numbers are meaningless")
    return ap


def check_args(a):
    if a.steps % a.segment_steps:
        raise SystemExit(f"--steps {a.steps} must be a multiple of --segment_steps {a.segment_steps}")
    if a.horizon > a.segment_steps:
        raise SystemExit(f"--horizon {a.horizon} must not exceed --segment_steps {a.segment_steps} (targets are per segment)")
    a.mode = "tuned" if a.tuned else "as_published"
    return a


def load_model():
    """The mjlab G1 as a mujoco.MjModel with sensors disabled, and its metadata."""
    import mujoco
    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f"mujoco {mujoco.__version__} cannot read {MODEL_FILE.name}: it is a MuJoCo {MUJOCO_VERSION} binary")
    mjm = mujoco.MjModel.from_binary_path(str(MODEL_FILE))
    meta = json.loads(META_FILE.read_text())
    assert (mjm.nq, mjm.nv, mjm.nu) == (NQ, NV, NU), (mjm.nq, mjm.nv, mjm.nu)
    assert abs(mjm.opt.timestep - TIMESTEP) < 1e-12 and meta["substeps_per_control_step"] == SUBSTEPS
    mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    return mjm, meta


def keyframe(mjm):
    import mujoco
    for name in ("knees_bent", "home"):
        k = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, name)
        if k >= 0:
            return name, np.array(mjm.key_qpos[k], np.float64)
    raise RuntimeError("the model has neither a knees_bent nor a home keyframe")


def workload(mjm, nworld, nsteps, seed=0):
    """The benchmark inputs as float32 numpy arrays: qpos0 (N, nq), qvel0 (N, nv), warm0 (N, nv), targets (T, N, nu) with
    T = nsteps, q_ref (nu,), plus the keyframe's name. The same call gives the same arrays in every environment."""
    rng = np.random.default_rng(seed)
    name, key = keyframe(mjm)
    q_ref = key[7:].copy()
    lo, hi = mjm.jnt_range[1:, 0], mjm.jnt_range[1:, 1]
    qpos0 = np.tile(key, (nworld, 1))
    qpos0[:, 7:] = np.clip(q_ref + SIGMA_Q0 * rng.standard_normal((nworld, NU)), lo + 0.02, hi - 0.02)
    qpos0[:, 2] += LIFT
    qvel0 = np.zeros((nworld, NV))
    qvel0[:, 6:] = SIGMA_V0 * rng.standard_normal((nworld, NU))
    e = np.zeros((nworld, NU))
    targets = np.empty((nsteps, nworld, NU), np.float32)
    clo, chi = mjm.actuator_ctrlrange[:, 0], mjm.actuator_ctrlrange[:, 1]
    for t in range(nsteps):
        e = RHO_U * e + np.sqrt(1.0 - RHO_U ** 2) * SIGMA_U * rng.standard_normal((nworld, NU))
        targets[t] = np.clip(q_ref + e, clo, chi)
    return dict(qpos0=qpos0.astype(np.float32), qvel0=qvel0.astype(np.float32),
                warm0=np.zeros((nworld, NV), np.float32), targets=targets, q_ref=q_ref.astype(np.float32), keyframe=name)


def foot_friction(mjm, meta):
    """The model's own friction of the foot-floor pairs (NablaEngine takes it as a per-world input)."""
    f = mjm.pair_friction[np.asarray(meta["foot_pairs"]), 0]
    assert np.allclose(f, f[0]), f
    return float(f[0])


def model_options(mjm):
    import mujoco
    o = mjm.opt
    return dict(solver=mujoco.mjtSolver(o.solver).name.replace("mjSOL_", ""),
                cone=mujoco.mjtCone(o.cone).name.replace("mjCONE_", ""),
                integrator=mujoco.mjtIntegrator(o.integrator).name.replace("mjINT_", ""),
                iterations=int(o.iterations), ls_iterations=int(o.ls_iterations), tolerance=float(o.tolerance),
                timestep=float(o.timestep))


VALID_SIZES = tuple(int(x) for x in (2 + (np.arange(19) % 2)) * (2 ** (np.arange(19) // 2 + 3)))  # 16, 24, 32, 48, ...


def valid_size(n):
    """The smallest of MuJoCo Warp's own capacity sizes (16, 24, 32, 48, 64, 96, 128, 192, ...) >= n."""
    for s in VALID_SIZES:
        if s >= n:
            return s
    return int(n)


def gpu_index():
    v = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    return v if v.isdigit() else None


def _smi(args):
    try:
        r = subprocess.run(["nvidia-smi", *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                           timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def gpu_info():
    """name, driver and memory [MiB] of the GPU this process uses (the first CUDA_VISIBLE_DEVICES entry; env.sh sets
    CUDA_DEVICE_ORDER=PCI_BUS_ID so that it is nvidia-smi's index)."""
    idx = gpu_index()
    out = _smi((["-i", idx] if idx else []) + ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return dict(gpu="", driver="", gpu_mem_total_mib="")
    p = [x.strip() for x in out.splitlines()[0].split(",")]
    return dict(gpu=p[0], driver=p[1], gpu_mem_total_mib=p[2])


def process_gpu_mem_mib():
    """GPU memory [MiB] this process holds according to the driver (nvidia-smi --query-compute-apps). The frameworks'
    memory pools do not shrink, so a reading after a pass is the pass's high-water mark of reserved memory (an upper
    bound of what it needed; the frameworks' own peak counters are reported next to it)."""
    out = _smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if out is None:
        return ""
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2 and p[0] == str(os.getpid()) and p[1].isdigit():
            return int(p[1])
    return ""


def packages(names):
    import importlib.metadata as md
    out = []
    for n in names:
        try:
            out.append(f"{n}={md.version(n)}")
        except md.PackageNotFoundError:
            out.append(f"{n}=absent")
    return ";".join(out)


def is_oom(exc):
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(k in msg for k in ("resource_exhausted", "out of memory", "failed to allocate", "outofmemory", "cuda error 2"))


class Csv:
    """One CSV line per measurement: stdout (prefix 'CSV,') and, with a path, appended to that file."""

    def __init__(self, path, base):
        self.path, self.base, self.header_done = (Path(path) if path else None), dict(base), False

    def _line(self, values):
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow(values)
        return buf.getvalue()

    def row(self, **kw):
        unknown = set(kw) - set(COLUMNS)
        if unknown:
            raise KeyError(f"not CSV columns: {sorted(unknown)}")
        rec = dict(self.base, **kw)
        rec.setdefault("utc", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        values = [rec.get(c, "") for c in COLUMNS]
        if not self.header_done:
            print("CSV," + self._line(COLUMNS), flush=True)
            self.header_done = True
        print("CSV," + self._line(values), flush=True)
        if self.path is not None:
            new = not self.path.exists() or self.path.stat().st_size == 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(COLUMNS)
                w.writerow(values)


def base_record(a, sim, mjm):
    """The columns every measurement of a run shares."""
    rec = dict(sim=sim, mode=a.mode, nworld=a.nworld, steps=a.steps, segment_steps=a.segment_steps, horizon=a.horizon,
               repeats=a.repeats, substeps=SUBSTEPS, timestep=TIMESTEP, precision="float32", seed=a.seed,
               python=platform.python_version())
    o = model_options(mjm)
    rec.update(solver=o["solver"], cone=o["cone"], integrator=o["integrator"])
    rec.update(gpu_info())
    return rec


def respawn(script, argv, jobs, drop=()):
    """Run the script once per job, one process each (so every measurement has its own peak memory and its own
    compiled programs): argv without the options named in drop (and their values), followed by the job's options.
    Returns the OR of the exit statuses."""
    keep, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x in drop:
            skip = True
            continue
        if any(x.startswith(o + "=") for o in drop):
            continue
        keep.append(x)
    rc = 0
    for job in jobs:
        say(f"[{Path(script).name}] starting: {' '.join(job)}")
        rc |= subprocess.call([sys.executable, str(Path(script).resolve()), *keep, *job])
    return rc


def median(values):
    return float(np.median(np.asarray(values, np.float64)))


def fmt(values, nd=4):
    return ";".join(f"{float(v):.{nd}f}" for v in values)


def say(*msg):
    print(*msg, file=sys.stderr, flush=True)

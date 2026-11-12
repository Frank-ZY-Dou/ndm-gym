"""RoboKit's MuJoCo Warp simulator (robokit.sim.MjWarpSim) with a world model of the MuJoCo model (worldmodels.mjcf) in place
of its physics step, for RoboKit's SimMPC. ``ndm_sim()`` returns a subclass whose ``step()`` first runs the MuJoCo Warp stages
the task's costs read (``derived``: kinematics, sensors, collision, ...) on the current state, so that derived fields lag the
state by one step as MuJoCo's do, then advances qpos, qvel and time by the world model of the step's length. Everything else is
the parent's: entities, sensors, resets, scenes, torch views, ``resized`` (which keeps the subclass and its world models).

On a GPU the world model's step is one torch CUDA graph over the simulator's own arrays (zero-copy views). Inside an enclosing
capture (SimMPC's per-step graphs) the graph is added to it as a child graph node, so a planning step stays one CUDA graph;
outside one it is replayed on Warp's stream. A staged planner steps several lengths (SimMPC writes ``opt.timestep`` before each
step): there is one world model per length, and ``planning()`` tells each simulator the length SimMPC sets by intercepting that
write at run time. On the device, a guard flags any step whose timestep differs from the length the world model was chosen for.
A length with no world model is an error: raised at once when the step runs eagerly, flagged on the device when it runs inside
a captured graph; ``check()`` raises on either flag. SimMPC warms up every step length it may use (1, 2 and 4 timesteps) while
it builds its graphs, so during that build a step of a length with no world model only captures the flag. RoboKit's files stay
unmodified."""

import contextlib  # no `from __future__ import annotations`: warp reads the kernels' annotations as objects
import ctypes
import types
import weakref

_CUDA = None
_KERNELS = None
_LIVE = weakref.WeakSet()  # every NdmSim, for planning()'s timestep hook
COMPILE_MODE = "max-autotune-no-cudagraphs"  # torch.compile's mode for NdmSim's world-model steps


def _driver():
    global _CUDA
    if _CUDA is None:
        lib = ctypes.CDLL("libcuda.so.1")
        for f in ("cuStreamGetCaptureInfo_v2", "cuGraphAddChildGraphNode", "cuStreamUpdateCaptureDependencies"):
            getattr(lib, f).restype = ctypes.c_int
        _CUDA = lib
    return _CUDA


def add_child_graph(stream: int, graph: int):
    """Add ``graph`` (a CUgraph, copied) to the graph being captured on ``stream`` (a CUstream handle) after the capture's current
    dependencies, and let the capture continue after it: cuGraphAddChildGraphNode, as replaying a graph inside a capture is not
    permitted."""
    lib = _driver()
    status, cid, parent = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p()
    deps, ndeps = ctypes.POINTER(ctypes.c_void_p)(), ctypes.c_size_t()
    r = lib.cuStreamGetCaptureInfo_v2(ctypes.c_void_p(stream), ctypes.byref(status), ctypes.byref(cid), ctypes.byref(parent),
                                      ctypes.byref(deps), ctypes.byref(ndeps))  # fmt: skip
    if r != 0 or status.value != 1:  # CU_STREAM_CAPTURE_STATUS_ACTIVE
        raise RuntimeError(f"add_child_graph: the stream is not capturing (CUresult {r}, capture status {status.value})")
    node = ctypes.c_void_p()
    r = lib.cuGraphAddChildGraphNode(ctypes.byref(node), parent, deps, ndeps, ctypes.c_void_p(graph))
    if r != 0:
        raise RuntimeError(f"add_child_graph: cuGraphAddChildGraphNode returned CUresult {r}")
    r = lib.cuStreamUpdateCaptureDependencies(ctypes.c_void_p(stream), ctypes.byref(node), ctypes.c_size_t(1), ctypes.c_uint(1))
    if r != 0:  # 1: CU_STREAM_SET_CAPTURE_DEPENDENCIES
        raise RuntimeError(f"add_child_graph: cuStreamUpdateCaptureDependencies returned CUresult {r}")


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        import warp as wp

        @wp.kernel
        def mark(code: int, flag: wp.array(dtype=int)):
            flag[0] = code

        @wp.kernel
        def guard(timestep: wp.array(dtype=float), substeps: int, dt: float, bad: wp.array(dtype=int)):
            k = wp.tid()
            step = timestep[0]
            if timestep.shape[0] > 1:
                step = timestep[k]
            if wp.abs(step * float(substeps) - dt) > 1e-6 * dt:
                bad[0] = 1

        _KERNELS = (guard, mark)
    return _KERNELS


def ndm_sim():
    """The world-model subclass of robokit.sim.MjWarpSim: ``NdmSim(model, nworld, ..., ndm={step length in s: world model},
    derived=("kinematics",))``. The world models are ``MjcfWorldModel``s or their ``inference_copy``s on the simulator's device;
    ``derived`` names MuJoCo Warp stages (``mujoco_warp.kinematics``, ``transmission``, ``sensor_pos``, ``collision``, ...) run in
    that order, or ``actuator_velocity``: the actuators' velocities from qvel and the moments ``transmission`` set (MuJoCo Warp's
    own kernel, the first part of ``fwd_velocity``, without its passive forces and inverse dynamics), which actuator-velocity
    sensors read."""
    import mujoco_warp as mjw
    import torch
    import warp as wp
    from robokit.sim import MjWarpSim

    class NdmSim(MjWarpSim):
        """MjWarpSim whose step is a world model's: ``derived`` stages on the current state, then the world model of the step.
        ``physics``: step lengths (s) that run MuJoCo Warp's own step instead, a hybrid planner (e.g. a fine first stage on the
        physics, the coarse rest on a world model); every other length without a world model stays an error."""

        def __init__(self, *args, ndm: dict, derived=("kinematics",), compile: bool = True, physics=(), **kwargs):  # noqa: A002
            super().__init__(*args, **kwargs)
            bad = [d for d in derived if d != "actuator_velocity" and not callable(getattr(mjw, d, None))]
            if bad:
                raise ValueError(f"NdmSim: not MuJoCo Warp stages: {bad}")
            if self.substeps > 1 and self.mjm.nsensordata > 0:
                raise ValueError("NdmSim: one world-model step stands for all substeps, so per-substep sensor data cannot be filled")
            if compile and not self.device.is_cuda:
                raise ValueError("NdmSim: compile=True needs a GPU; pass compile=False on Warp's CPU device")
            self.ndm, self.derived, self.compile = {round(float(k), 9): v for k, v in ndm.items()}, tuple(derived), compile
            off = {k: m.lay.dt for k, m in self.ndm.items() if abs(m.lay.dt - k) > 1e-9}
            if off:
                raise ValueError(f"NdmSim: world models given for these step lengths step others: {off} (s: s)")
            self.physics = tuple(sorted({round(float(x), 9) for x in physics}))
            both = sorted(set(self.physics) & set(self.ndm))
            if both:
                raise ValueError(f"NdmSim: step lengths {both} s have a world model and are also to run the physics")
            self.host_dt = round(float(self.mjm.opt.timestep) * self.substeps, 9)  # the length planning() last saw set
            d = self._d
            self.views = tuple(wp.to_torch(a) for a in (d.qpos, d.qvel, d.ctrl, d.time))
            with wp.ScopedDevice(self.device):
                self.bad = wp.zeros(1, dtype=int)  # a step whose timestep was not its world model's
                self.missing = wp.zeros(1, dtype=int)  # a captured step of a length with no world model, in microseconds
            self.building = False  # set by planning() while SimMPC builds its graphs
            if self.device.is_cuda:  # one CUDA graph per step length; on the CPU the step runs eagerly
                self.graphs = {dt: self._build(dt, m) for dt, m in self.ndm.items()}
            else:
                self.bodies = {dt: self._body(dt, m) for dt, m in self.ndm.items()}
            _LIVE.add(self)

        def resized(self, nworld: int) -> "NdmSim":
            scenes = self._args["model"]
            n = nworld if not isinstance(scenes, (list, tuple)) else [nworld // len(scenes)] * len(scenes)
            return type(self)(nworld=n, ndm=self.ndm, derived=self.derived, compile=self.compile, physics=self.physics, **self._args)

        def _body(self, dt: float, m):
            q, v, u, t = self.views
            fn = torch.compile(m.step, mode=COMPILE_MODE, dynamic=False) if self.compile else m.step

            def body():
                q1, v1 = fn(q, v, u)
                q.copy_(q1)
                v.copy_(v1)
                t.add_(dt)

            return body

        def _build(self, dt: float, m):
            """The world model's step of length ``dt`` as a torch CUDA graph on the simulator's arrays (warmed up on copies, so
            the state is not touched; the capture runs nothing)."""
            q, v, u, t = self.views
            saved = tuple(x.clone() for x in self.views)
            body = self._body(dt, m)
            side = torch.cuda.Stream(device=q.device)
            side.wait_stream(torch.cuda.current_stream(q.device))
            with torch.no_grad(), torch.cuda.stream(side):
                for _ in range(3):  # compilation and the allocator's warm-up
                    body()
                for x, y in zip(self.views, saved):
                    x.copy_(y)
            torch.cuda.current_stream(q.device).wait_stream(side)
            torch.cuda.synchronize(q.device)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.no_grad(), torch.cuda.graph(graph):
                body()
            return graph

        def step(self):
            dt = self.host_dt
            guard, mark = _kernels()
            if dt in self.physics:  # declared: this length runs MuJoCo Warp's own step
                return super().step()
            if dt not in self.ndm:
                with wp.ScopedDevice(self.device):
                    stream = wp.get_stream() if self.device.is_cuda else None
                    if stream is not None and stream.is_capturing:  # a graph SimMPC may never replay: flag it if it does
                        wp.launch(mark, dim=1, inputs=[int(round(dt * 1e6))], outputs=[self.missing])
                        return
                if self.building:  # SimMPC's warm-up run of a step length its plans may not use
                    return
                raise ValueError(f"NdmSim: a step of {dt} s; world models for {sorted(self.ndm)} s")
            with wp.ScopedDevice(self.device):
                wp.launch(guard, dim=self.nworld, inputs=[self._m.opt.timestep, self.substeps, dt], outputs=[self.bad])
                if self.derived:
                    self._launch("ndm_derived", lambda: [self._stage(s) for s in self.derived])
                if self.mjm.nsensordata:  # the sensors of the state before the step, as MjWarpSim's single substep leaves them
                    wp.copy(self.data.substep_sensordata, self._d.sensordata)
                stream = wp.get_stream() if self.device.is_cuda else None
                if stream is None:  # Warp's CPU device: the world model eagerly
                    with torch.no_grad():
                        self.bodies[dt]()
                elif stream.is_capturing:
                    add_child_graph(stream.cuda_stream, self.graphs[dt].raw_cuda_graph())
                else:
                    with torch.cuda.stream(wp.stream_to_torch(stream)):
                        self.graphs[dt].replay()

        def _stage(self, name: str):
            if name != "actuator_velocity":
                return getattr(mjw, name)(self._m, self._d)
            from mujoco_warp._src.forward import _actuator_velocity  # as fwd_velocity launches it (mujoco_warp 4.0.1)

            d = self._d
            wp.launch(_actuator_velocity, dim=(d.nworld, self.mjm.nu), inputs=[d.qvel, d.moment_rownnz, d.moment_rowadr, d.moment_colind,
                                                                             d.actuator_moment], outputs=[d.actuator_velocity])  # fmt: skip

        def check(self):
            """Raise if a step ran with a timestep other than its world model's, or a captured step of a length with no world
            model ran (a host sync)."""
            if int(self.bad.numpy()[0]):
                raise RuntimeError("NdmSim: a step ran with a timestep its world model was not chosen for (set it through planning())")
            missing = int(self.missing.numpy()[0])
            if missing:
                raise RuntimeError(f"NdmSim: a plan stepped {missing * 1e-6:g} s; world models for {sorted(self.ndm)} s")

    return NdmSim


@contextlib.contextmanager
def planning():
    """While active, SimMPC's write of a step's timestep (its ``_set_value_kernel`` launch into a simulator's ``opt.timestep``)
    also sets that NdmSim's host-side step length, so that the step it captures or runs next uses the world model of that
    length. ``warp.launch`` is wrapped for the duration (the module objects stay as they are)."""
    import warp
    from robokit.helpers.sim_mpc import sim_mpc as mod

    real, build = warp.launch, mod.SimMPC._build

    def _build(self, *args, **kwargs):
        for sim in list(_LIVE):
            sim.building = True
        try:
            return build(self, *args, **kwargs)
        finally:
            for sim in list(_LIVE):
                sim.building = False

    def launch(kernel, *args, **kwargs):
        if kernel is mod._set_value_kernel:
            inputs = kwargs.get("inputs", args[1] if len(args) > 1 else None)
            outputs = kwargs.get("outputs", args[2] if len(args) > 2 else None)
            for sim in list(_LIVE):
                if outputs and outputs[0] is sim._m.opt.timestep:
                    sim.host_dt = round(float(inputs[0]) * sim.substeps, 9)
        return real(kernel, *args, **kwargs)

    warp.launch, mod.SimMPC._build = launch, _build
    try:
        yield
    finally:
        warp.launch, mod.SimMPC._build = real, build


def load_models(paths: dict, model, device, dtype=None) -> dict:
    """{step length: inference copy} from fit_mjcf --save files ({step length: path}), each checked against its length."""
    from ..wmbench.speed_mjcf import load_ndm
    from ..worldmodels.mjcf import MjcfLayout, inference_copy

    out = {}
    for dt, path in paths.items():
        wm = load_ndm(MjcfLayout.from_model(model), types.SimpleNamespace(ckpt=path), device)
        if abs(wm.lay.dt - float(dt)) > 1e-9:
            raise ValueError(f"load_models: {path} steps {wm.lay.dt} s, given for {dt} s")
        out[float(dt)] = inference_copy(wm, dtype)
    return out

"""Persistent native Door worlds with one physical control step per collection.

Shares the immutable model, keeps separate MjData, and preserves native solver
options (including noslip). This serial correctness path makes no throughput or
GPU equivalence claim. Caller must budget MjData memory before large creation.
"""

import numpy as np

from .adroit_door import DoorPhysics


class NativeDoorCollector:
    def __init__(self, model, reference, worlds, *, physics_cls=DoorPhysics):
        if type(worlds) is not int or worlds < 1:
            raise ValueError('worlds must be a positive integer')
        self.worlds = tuple(physics_cls(model, reference) for _ in range(worlds))
        self.transitions = 0
        self.failed = False

    def observe(self):
        """Policy inputs after any previous terminal worlds were reset."""
        q, v = zip(*(world.state() for world in self.worlds))
        return dict(qpos=np.stack(q), qvel=np.stack(v),
                    frames=np.array([world.frame for world in self.worlds], dtype=np.int64))

    def step(self, actions):
        """Return pre/post physics states and boundaries, then reset done worlds.

        The returned next state is always the terminal physical state, never
        the reset state. Windows must be split at ``done``. A physical failure
        poisons the collector: a partially advanced batch cannot be retried.
        """
        if self.failed:
            raise RuntimeError('Collector failed; reconstruct from a valid checkpoint')
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (len(self.worlds), self.worlds[0].model.nu) or not np.isfinite(actions).all():
            raise ValueError('Expected finite actions (worlds, 28)')
        actions = np.clip(actions, -1, 1)
        before = self.observe()
        try:
            results = [world.step(action) for world, action in zip(self.worlds, actions)]
            q, v = zip(*(state for state, _ in results))
            done = np.array([terminal for _, terminal in results], dtype=bool)
            batch = dict(**before, actions=actions.copy(), next_qpos=np.stack(q),
                         next_qvel=np.stack(v), done=done)
            for world, terminal in zip(self.worlds, done):
                if terminal:
                    world.reset()
        except Exception:
            self.failed = True
            raise
        self.transitions += len(self.worlds)
        return batch


class PackedNativeDoorCollector:
    """Serial worlds multiplexed through one native solver arena.

    Saves mjSTATE_INTEGRATION, including warmstart, controls and applied forces;
    saving qpos/qvel alone would not preserve native stepping. No mj_forward is
    inserted between restore and mj_step. Must be benchmarked before training.
    """

    def __init__(self, model, reference, worlds, *, physics_cls=DoorPhysics):
        import mujoco

        if type(worlds) is not int or worlds < 1:
            raise ValueError('worlds must be a positive integer')
        self.sim = physics_cls(model, reference)
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        initial = np.empty(mujoco.mj_stateSize(model, self.spec))
        mujoco.mj_getState(model, self.sim.data, initial, self.spec)
        self.states = np.repeat(initial[None], worlds, axis=0)
        self.qpos = np.repeat(reference.qpos[:1], worlds, axis=0).copy()
        self.qvel = np.repeat(reference.qvel[:1], worlds, axis=0).copy()
        self.frames = np.zeros(worlds, dtype=np.int64)
        self.transitions = 0
        self.failed = False

    def observe(self):
        return dict(qpos=self.qpos.copy(), qvel=self.qvel.copy(), frames=self.frames.copy())

    def step(self, actions):
        import mujoco

        if self.failed:
            raise RuntimeError('Collector failed; reconstruct from a valid checkpoint')
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (len(self.states), self.sim.model.nu) or not np.isfinite(actions).all():
            raise ValueError('Expected finite actions (worlds, 28)')
        actions = np.clip(actions, -1, 1)
        before = self.observe()
        next_q, next_v = np.empty_like(self.qpos), np.empty_like(self.qvel)
        done = np.zeros(len(self.states), dtype=bool)
        try:
            for i, action in enumerate(actions):
                mujoco.mj_setState(self.sim.model, self.sim.data, self.states[i], self.spec)
                self.sim.frame, self.sim.done = int(self.frames[i]), False
                (next_q[i], next_v[i]), done[i] = self.sim.step(action)
                if done[i]:
                    self.sim.reset()
                self.qpos[i], self.qvel[i] = self.sim.state()
                self.frames[i] = self.sim.frame
                mujoco.mj_getState(self.sim.model, self.sim.data, self.states[i], self.spec)
        except Exception:
            self.failed = True
            raise
        self.transitions += len(self.states)
        return dict(**before, actions=actions.copy(), next_qpos=next_q,
                    next_qvel=next_v, done=done)


class CompiledPackedDoorCollector(PackedNativeDoorCollector):
    """Same integration-state algorithm with the world loop outside Python.

    The supplied trusted library must be built with tools/build_door_native.py
    using this runtime's MuJoCo. CDLL releases the GIL for the whole chunk.
    """

    def __init__(self, model, reference, worlds, library, *, physics_cls=DoorPhysics):
        import ctypes
        import mujoco
        super().__init__(model, reference, worlds, physics_cls=physics_cls)
        self.library = ctypes.CDLL(str(library))
        if self.library.door_mujoco_version() != mujoco.mj_version():
            raise ValueError('Native library MuJoCo version mismatch')
        ptr = ctypes.c_void_p
        self.native_step = self.library.door_batch
        self.native_step.argtypes = [ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ptr]*12
        self.native_step.restype = ctypes.c_int
        self.initial_q = np.ascontiguousarray(reference.qpos[0], dtype=np.float64)
        self.initial_v = np.ascontiguousarray(reference.qvel[0], dtype=np.float64)
        self.states = np.ascontiguousarray(self.states, dtype=np.float64)
        self.qpos = np.ascontiguousarray(self.qpos, dtype=np.float64)
        self.qvel = np.ascontiguousarray(self.qvel, dtype=np.float64)
        self.frames = np.ascontiguousarray(self.frames, dtype=np.int64)

    def step(self, actions):
        if self.failed:
            raise RuntimeError('Collector failed; reconstruct from a valid checkpoint')
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (len(self.states), self.sim.model.nu) or not np.isfinite(actions).all():
            raise ValueError('Expected finite actions (worlds, 28)')
        actions = np.ascontiguousarray(np.clip(actions, -1, 1))
        before = self.observe()
        next_q, next_v = np.empty_like(self.qpos), np.empty_like(self.qvel)
        done = np.empty(len(self.states), dtype=np.uint8)
        arrays = (self.states, self.frames, actions, self.sim.mid, self.sim.half,
                  self.initial_q, self.initial_v, self.qpos, self.qvel, next_q, next_v, done)
        result = self.native_step(self.sim.model._address, self.sim.data._address,
                                  len(self.states), self.sim.frame_skip,
                                  self.sim.reference.frames - 1,
                                  *(x.ctypes.data for x in arrays))
        if result:
            self.failed = True
            raise RuntimeError(f'Native Door batch rejected transition (code {result})')
        self.transitions += len(self.states)
        return dict(**before, actions=actions, next_qpos=next_q, next_qvel=next_v,
                    done=done.astype(bool))


class ParallelNativeDoorCollector:
    """Persistent bounded chunk workers, one solver arena per worker.

    Policy inference belongs to the caller and consumes/produces whole batches.
    Each chunk retains mjSTATE_INTEGRATION across calls, including warmstart.
    Call close (or use a context manager) before releasing the collector.
    """

    def __init__(self, model, reference, worlds, workers=1, native_library=None, *, physics_cls=DoorPhysics):
        from concurrent.futures import ThreadPoolExecutor

        if type(worlds) is not int or worlds < 1:
            raise ValueError('worlds must be a positive integer')
        if type(workers) is not int or not 1 <= workers <= worlds:
            raise ValueError('workers must be between one and worlds')
        factory = (lambda m, r, n: PackedNativeDoorCollector(m, r, n, physics_cls=physics_cls)) if native_library is None else (
            lambda m, r, n: CompiledPackedDoorCollector(m, r, n, native_library, physics_cls=physics_cls))
        self.chunks = tuple(factory(model, reference, len(indices))
                            for indices in np.array_split(np.arange(worlds), workers))
        self.slices = tuple(np.cumsum([0] + [len(c.states) for c in self.chunks]))
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='door-native')
        self.previous_actions = np.zeros((worlds, model.nu))
        self.transitions = 0
        self.failed = False
        self.closed = False

    def observe(self):
        observations = [c.observe() for c in self.chunks]
        return {key: np.concatenate([o[key] for o in observations]) for key in observations[0]}

    def stagger_from_physical_replay(self, seed=0):
        """Initialize phases from actual native replay, including solver warmstarts.

        Only called before collection. Unlike assigning kinematic targets, every
        snapshot is the result of executing reference controls from frame zero.
        This is initialization work and is reported separately from training.
        """
        import mujoco
        if self.transitions or self.closed or self.failed:
            raise RuntimeError('Staggering requires a new collector')
        first = self.chunks[0]
        sim = type(first.sim)(first.sim.model, first.sim.reference)
        ref = sim.reference
        snapshots, positions, velocities = [], [], []
        for action in ref.actions:
            integration = np.empty(first.states.shape[1])
            mujoco.mj_getState(sim.model, sim.data, integration, first.spec)
            snapshots.append(integration)
            q, v = sim.state()
            positions.append(q)
            velocities.append(v)
            sim.step(action)
        phases = np.arange(len(self.previous_actions)) % len(ref.actions)
        np.random.default_rng(seed).shuffle(phases)
        self.load_state_dict(dict(states=np.asarray(snapshots)[phases],
                                  qpos=np.asarray(positions)[phases], qvel=np.asarray(velocities)[phases],
                                  frames=phases.astype(np.int64), transitions=0,
                                  previous_actions=np.where((phases > 0)[:, None],
                                                            ref.actions[(phases-1).clip(0)].clip(-1, 1), 0.)))

    def state_dict(self):
        if self.closed or self.failed:
            raise RuntimeError('Cannot snapshot an invalid collector')
        return dict(states=np.concatenate([c.states for c in self.chunks]),
                    **self.observe(), previous_actions=self.previous_actions.copy(),
                    transitions=self.transitions)

    def load_state_dict(self, state):
        """Restore only after the caller has checked model/reference identity."""
        if self.closed or self.failed:
            raise RuntimeError('Cannot restore an invalid collector')
        n = len(self.previous_actions)
        shapes = dict(states=(n, self.chunks[0].states.shape[1]), qpos=(n, self.chunks[0].sim.model.nq),
                      qvel=(n, self.chunks[0].sim.model.nv), frames=(n,),
                      previous_actions=(n, self.chunks[0].sim.model.nu))
        for key, shape in shapes.items():
            if state[key].shape != shape or not np.isfinite(state[key]).all():
                raise ValueError(f'Invalid collector checkpoint: {key}')
        frames = state['frames']
        if (frames.dtype != np.int64 or np.any(frames < 0)
                or np.any(frames >= self.chunks[0].sim.reference.frames - 1)
                or type(state['transitions']) is not int or state['transitions'] < 0
                or np.any(np.abs(state['previous_actions']) > 1)):
            raise ValueError('Invalid collector boundaries/count/actions')
        for c, a, b in zip(self.chunks, self.slices[:-1], self.slices[1:]):
            for key in ('states', 'qpos', 'qvel', 'frames'):
                getattr(c, key)[:] = state[key][a:b]
        self.previous_actions[:] = state['previous_actions']
        self.transitions = state['transitions']

    def step(self, actions):
        if self.failed or self.closed:
            raise RuntimeError('Collector failed or closed; reconstruct before stepping')
        actions = np.asarray(actions, dtype=float)
        if actions.shape != self.previous_actions.shape or not np.isfinite(actions).all():
            raise ValueError('Expected finite actions (worlds, 28)')
        actions = np.clip(actions, -1, 1).copy()
        previous = self.previous_actions.copy()
        futures = [self.pool.submit(c.step, actions[a:b])
                   for c, a, b in zip(self.chunks, self.slices[:-1], self.slices[1:])]
        # Drain every worker before exposing an error: no background mutation remains.
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            self.failed = True
            raise errors[0]
        batch = {key: np.concatenate([r[key] for r in results]) for key in results[0]}
        batch['previous_actions'] = previous
        self.previous_actions[:] = actions
        self.previous_actions[batch['done']] = 0
        self.transitions += len(actions)
        return batch

    def close(self):
        self.pool.shutdown(wait=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

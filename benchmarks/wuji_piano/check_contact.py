"""Check scripted piano notes and chords with Wuji hands in CPU MuJoCo and optionally on CUDA."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from piano_score import CONTROL_SUBSTEPS, KEY_COUNT, LOWEST_PITCH, PHYSICS_STEP

MIDDLE_C = 60
WHITE_KEY_PITCH = 0.0235
HOLD_STEPS = 10 * CONTROL_SUBSTEPS
WRIST_DOWN = -0.022

MELODY = (
    (0, 0),
    (0, WRIST_DOWN),
    (0, 0),
    (1, 0),
    (1, WRIST_DOWN),
    (1, 0),
    (2, 0),
    (2, WRIST_DOWN),
    (2, 0),
    (2, WRIST_DOWN),
    (2, 0),
)
CHORD = ((2, 0), (2, -0.024), (2, 0))


def mount_hand(piano: mujoco.MjSpec, hand_xml: Path, side: str) -> None:
    hand = mujoco.MjSpec.from_file(str(hand_xml))
    hand.option = piano.option
    for body in hand.bodies[1:]:
        body.gravcomp = 1

    # Center each index finger over middle C at zero lateral displacement.
    wrist_y = -0.05145 if side == "right" else -0.06605
    wrist = piano.worldbody.add_body(name=f"{side}_carrier", pos=[0.215, wrist_y, 0.13])
    for axis_name, axis, limits, control_range in (
        ("height", [0, 0, 1], [-0.045, 0.005], [-0.045, 0]),
        ("lateral", [0, 1, 0], [-0.01, 0.06], [0, 0.05]),
    ):
        name = f"{side}_{axis_name}"
        wrist.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis, range=limits)
        actuator = piano.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=control_range)
        actuator.set_to_position(kp=1800, kv=55)
    half_angle = np.deg2rad(35)
    mount = wrist.add_frame(quat=[np.cos(half_angle), 0, np.sin(half_angle), 0])
    piano.attach(hand, prefix=f"{side}/", frame=mount)


def build_scene(hand_xml: Path, keyboard_xml: Path, left_hand_xml: Path | None = None) -> mujoco.MjModel:
    piano = mujoco.MjSpec.from_file(str(keyboard_xml))
    piano.option.timestep = PHYSICS_STEP
    piano.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for key in range(KEY_COUNT):
        piano.body(f"key_{key}").gravcomp = 1

    mount_hand(piano, hand_xml, "right")
    if left_hand_xml is not None:
        mount_hand(piano, left_hand_xml, "left")
    return piano.compile()


def scripted_controls(model: mujoco.MjModel, targets) -> np.ndarray:
    controls = np.zeros((len(targets), model.nu), dtype=np.float32)
    for stage, (key_offset, height) in enumerate(targets):
        controls[stage, model.actuator("right_lateral").id] = key_offset * WHITE_KEY_PITCH
        controls[stage, model.actuator("right_height").id] = height
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_height") >= 0:
            controls[stage, model.actuator("left_height").id] = WRIST_DOWN if height else 0
    return controls


def initial_data(model: mujoco.MjModel, control: np.ndarray) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.ctrl[:] = control
    # Start over the first note. For the chord the hands have distinct starting keys.
    data.qpos[model.joint("right_lateral").qposadr[0]] = control[model.actuator("right_lateral").id]
    mujoco.mj_forward(model, data)
    return data


def check_strikes(model: mujoco.MjModel, controls, expected, chord: bool = False) -> list[tuple[int, str, float]]:
    data = initial_data(model, controls[0])
    keys = [model.joint(f"key_{i}_joint") for i in range(KEY_COUNT)]
    key_positions = np.array([joint.qposadr[0] for joint in keys])
    key_limits = np.array([joint.range[1] for joint in keys])
    pitches = {pitch for pitch, _ in expected}
    key_geoms = {model.geom(f"key_{pitch - LOWEST_PITCH}_geom").id for pitch in pitches}
    pressed = np.zeros(KEY_COUNT, dtype=bool)
    events = []
    contact_bodies = set()

    for control in controls:
        for _ in range(10):
            data.ctrl[:] = control
            for _ in range(CONTROL_SUBSTEPS):
                mujoco.mj_step(model, data)
                travel = data.qpos[key_positions] / key_limits
                next_pressed = np.where(pressed, travel > 0.35, travel >= 0.75)
                for key in np.flatnonzero(pressed != next_pressed):
                    action = "press" if next_pressed[key] else "release"
                    events.append((key + LOWEST_PITCH, action, data.time))
                    print(f"{data.time:.3f}s  MIDI {key + LOWEST_PITCH}  {action}")
                pressed = next_pressed
                for contact in data.contact:
                    for geom, other in (contact.geom, contact.geom[::-1]):
                        if geom in key_geoms:
                            contact_bodies.add(model.body(model.geom_bodyid[other]).name)

    if (
        not notes_match(events, expected, chord)
        or not contact_bodies
        or any(not body.startswith(("right/r_index_finger_", "left/l_index_finger_")) for body in contact_bodies)
    ):
        raise RuntimeError(f"Unexpected events or contacts: {events}, {sorted(contact_bodies)}")
    if not np.isfinite(data.qpos).all() or data.warning.number.any():
        raise RuntimeError(f"Simulation warnings: {data.warning.number}")
    if np.max(np.abs(data.qpos[key_positions])) > 0.001:
        raise RuntimeError("Keys did not return to rest")
    if chord:
        check_chord_overlap(events)
    print("CPU: exactly the expected notes, index-finger contact only, all keys returned to rest.")
    return events


def notes_match(events, expected, chord: bool) -> bool:
    notes = [(pitch, action) for pitch, action, _ in events]
    if chord:
        return len(notes) == 4 and set(notes[:2]) == set(expected[:2]) and set(notes[2:]) == set(expected[2:])
    return notes == expected


def check_chord_overlap(events) -> None:
    if abs(events[1][2] - events[0][2]) > 4 * PHYSICS_STEP + 1e-12:
        raise RuntimeError(f"Chord onsets are more than 20 ms apart: {events}")


def check_cuda(model: mujoco.MjModel, staged_controls, expected, worlds: int, chord: bool = False) -> None:
    import mujoco_warp as mjw
    import warp as wp

    keys = [model.joint(f"key_{i}_joint") for i in range(KEY_COUNT)]
    key_positions = np.array([joint.qposadr[0] for joint in keys])
    key_limits = np.array([joint.range[1] for joint in keys])
    pressed = np.zeros((worlds, KEY_COUNT), dtype=bool)
    events = [[] for _ in range(worlds)]
    controls = np.zeros((worlds, model.nu), dtype=np.float32)
    with wp.ScopedDevice("cuda:0"):
        gpu_model = mjw.put_model(model)
        data = mjw.put_data(model, initial_data(model, staged_controls[0]), nworld=worlds, nconmax=128, njmax=512)
        for stage, control in enumerate(staged_controls):
            controls[:] = control
            data.ctrl.assign(controls)
            for step in range(HOLD_STEPS):
                mjw.step(gpu_model, data)
                travel = data.qpos.numpy()[:, key_positions] / key_limits
                next_pressed = np.where(pressed, travel > 0.35, travel >= 0.75)
                time = (stage * HOLD_STEPS + step + 1) * PHYSICS_STEP
                for world, key in np.argwhere(pressed != next_pressed):
                    action = "press" if next_pressed[world, key] else "release"
                    events[world].append((key + LOWEST_PITCH, action, time))
                pressed = next_pressed
        qpos = data.qpos.numpy()

    expected_notes = [(pitch, action) for pitch, action, _ in expected]
    max_error = 0.0
    for world, actual in enumerate(events):
        if not notes_match(actual, expected_notes, chord):
            raise RuntimeError(f"CUDA world {world} produced unexpected note events: {actual}")
        if chord:
            check_chord_overlap(actual)
        cpu_times = {(pitch, action): time for pitch, action, time in expected} if chord else None
        for (pitch, action, gpu_time), (_, _, cpu_time) in zip(actual, expected):
            if cpu_times is not None:
                cpu_time = cpu_times[pitch, action]
            max_error = max(max_error, abs(gpu_time - cpu_time))
    if not np.isfinite(qpos).all() or np.max(np.abs(qpos[:, key_positions])) > 0.001:
        raise RuntimeError("CUDA rollout produced non-finite state or keys did not return to rest")
    if max_error > 2 * PHYSICS_STEP + 1e-12:
        raise RuntimeError(f"CUDA event timing differed from CPU by {max_error * 1000:.1f} ms")
    print(f"CUDA: {worlds} worlds matched CPU notes, max timing difference {max_error * 1000:.1f} ms.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, type=Path, help="Right Wuji Hand 2 beta2 MJCF")
    parser.add_argument("--keyboard", required=True, type=Path, help="Eden 88-key piano MJCF")
    parser.add_argument("--cuda-worlds", type=int, default=0, help="Also replay this many worlds on cuda:0")
    parser.add_argument("--left-hand", type=Path, help="Also check a two-hand C+E chord with this left beta2 MJCF")
    args = parser.parse_args()
    if args.cuda_worlds < 0:
        parser.error("--cuda-worlds must be nonnegative")
    model = build_scene(args.hand, args.keyboard)
    melody = [(pitch, action) for pitch in (60, 62, 64, 64) for action in ("press", "release")]
    controls = scripted_controls(model, MELODY)
    events = check_strikes(model, controls, melody)
    if args.cuda_worlds:
        check_cuda(model, controls, events, args.cuda_worlds)

    if args.left_hand:
        model = build_scene(args.hand, args.keyboard, args.left_hand)
        controls = scripted_controls(model, CHORD)
        chord = [(pitch, action) for action in ("press", "release") for pitch in (60, 64)]
        events = check_strikes(model, controls, chord, chord=True)
        if args.cuda_worlds:
            check_cuda(model, controls, events, args.cuda_worlds, chord=True)


if __name__ == "__main__":
    main()

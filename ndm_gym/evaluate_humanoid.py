"""Evaluate a humanoid tracking policy (train_humanoid's last.pt), or the reference pose alone as the position targets, on
every clip in the simulator: ndm_gym.evaluate_quadruped's evaluation (its options, rollouts from perturbed starts, pushes,
open-loop reference targets, ``--ckpts`` time to a target) with the humanoid's task (tasks.humanoid_tracking).

    python -m ndm_gym.evaluate_humanoid --ckpt logs/<name>/last.pt --out logs/<name>/evaluation --push 1 2
    python -m ndm_gym.evaluate_humanoid --open_loop --robot h1 --out logs/h1_open_loop
    python -m ndm_gym.evaluate_humanoid --ckpts logs/<name> --out logs/<name>/evaluation --success 0.99
    python -m ndm_gym.evaluate_humanoid --run logs/<name> --out logs/<name>/evaluation --num_envs 4 --push 1 2   # all in one pass
    python -m ndm_gym.evaluate_humanoid --run logs/<name> --out logs/<name>/windows --window_s 10                # windows, fast

``--run``: the last checkpoint (with the reference targets and the pushes) and every kept checkpoint as blocks of one
simulator, one pass over the clips. ``--window_s``: every clip cut into windows of that length, each rolled out from the
reference's state at its start (perturbed), all at once: window success and window return, for scanning checkpoints and
diagnosis, not the benchmark's whole-clip metrics. The control step runs as one CUDA graph (``--graph``) with the policy
inputs, errors and rewards compiled (``--compile``), both on by default here (``--eager``: neither).

The errors are the G1 evaluation's (ndm_gym.evaluate): MPJPE of the tracked bodies in the world (``mpjpe_g_mm``) and about
the pelvis (``mpjpe_l_mm``), the pelvis' position (``root_mm``) and rotation, and the joints' mean absolute error
(``joint_rad``, ``joint_deg``); means over the rollouts and the steps before a rollout's termination. The return, the
success and the failed clips are evaluate_quadruped's (the G1's, over several rollouts per clip). ``--robot`` picks the
robot without a checkpoint (``--open_loop``; default the H1); with a checkpoint the robot is its own and another is refused. The termination, and so the success, is the
task's: the G1's (the pelvis' height within 0.25 m and its tilt within 0.8 of the reference's), with ``--term_pos`` also
the pelvis within that distance of the reference's (m; default none), whatever the policy trained with. The servos accept
the position targets the policy trained with (its checkpoint's ``target_range``, robots.humanoid.TARGET_RANGES; a
checkpoint from before that option: the joints' ranges); ``--target_range`` picks them without a checkpoint (``--open_loop``).
The servos' gains (``--gains``, robots.humanoid.with_gains) are the checkpoint's. The policy follows the reference it trained
with (its ``--ref_ground`` height correction, tasks.humanoid_tracking.ground_reference: its goal, its targets' feed-forward
pose and the starts); ``--score_ground {none,clip,stance}`` measures the return, the errors and the termination against
another correction of the same clips (default the checkpoint's; ``--score_ground none``: the retarget's own heights), so that
a policy trained on a corrected reference is also scored against the original one.
"""

from __future__ import annotations

import sys


def humanoid_setup(targs, clips, dev: str, robot: str | None = None, term_pos: float = float("inf"), target_range: str | None = None,
                   score_ground: str | None = None):
    """evaluate_quadruped.main's ``setup``: the humanoid's scene, kinematics, task and clip names (train_humanoid.build with a
    checkpoint's robot, servos, accepted position targets and feet), its agents and its simulator's factory.
    ``target_range``: the servos' accepted targets without a checkpoint (--open_loop; default the joints' ranges); with one,
    the checkpoint's (a checkpoint from before the option: the joints' ranges), and a different value is refused. The gains
    and the reference's height correction are the checkpoint's (before those options: the robot's, none); ``score_ground``:
    the height correction of the reference the policy is scored against (None: the checkpoint's)."""
    from ndm_gym.algos.humanoid_algos import HUMANOID_ALGOS
    from ndm_gym.train_humanoid import build, make_sim

    g = (lambda k, d: getattr(targs, k, d)) if targs is not None else (lambda k, d: d)
    trained_robot = g("robot", robot or "h1")
    if robot is not None and robot != trained_robot:
        raise SystemExit(f"evaluate_humanoid: --robot {robot}: the checkpoint trained the {trained_robot}")
    trained = g("target_range", target_range or "joint")
    if target_range is not None and target_range != trained:
        raise SystemExit(f"evaluate_humanoid: --target_range {target_range}: the checkpoint trained with {trained}")
    model, kin, task, names = build(trained_robot, clips, dev, kp_scale=g("kp_scale", 1.0), kd_scale=g("kd_scale", 1.0), term_pos=term_pos,
                                   goal_anchor=g("goal_anchor", "robot"), foot=g("foot", "mesh"),
                                   ankle_effort_scale=g("ankle_effort_scale", 1.0), target_range=trained, gains=g("gains", "robot"),
                                   ref_ground=g("ref_ground", "none"), score_ground=score_ground)
    return model, kin, task, names, HUMANOID_ALGOS, lambda worlds, joint_noise=0.05: make_sim(model, task, worlds, dev, joint_noise)


def main():
    from ndm_gym.evaluate_quadruped import main as evaluate
    from ndm_gym.robots.humanoid import TARGET_RANGES
    from ndm_gym.tasks.humanoid_tracking import REF_GROUNDS

    opts = {"--robot": None, "--term_pos": "inf", "--target_range": None, "--score_ground": None}
    for name in opts:  # taken out before evaluate_quadruped's parser
        if name in sys.argv:
            i = sys.argv.index(name)
            opts[name] = sys.argv[i + 1]
            del sys.argv[i : i + 2]
    robot, term_pos, target_range, score_ground = opts["--robot"], float(opts["--term_pos"]), opts["--target_range"], opts["--score_ground"]
    if target_range is not None and target_range not in TARGET_RANGES:
        raise SystemExit(f"evaluate_humanoid: --target_range {target_range!r}: one of {TARGET_RANGES}")
    if score_ground is not None and score_ground not in REF_GROUNDS:
        raise SystemExit(f"evaluate_humanoid: --score_ground {score_ground!r}: one of {REF_GROUNDS}")
    if term_pos <= 0:
        raise SystemExit("evaluate_humanoid: --term_pos must be positive")
    if "--eager" in sys.argv:
        if "--graph" in sys.argv or "--compile" in sys.argv:
            raise SystemExit("evaluate_humanoid: --eager is without --graph and --compile")
        sys.argv.remove("--eager")
    else:
        sys.argv += [o for o in ("--graph", "--compile") if o not in sys.argv]
    evaluate(lambda targs, clips, dev: humanoid_setup(targs, clips, dev, robot, term_pos, target_range, score_ground), __doc__.split("\n")[0])


if __name__ == "__main__":
    main()

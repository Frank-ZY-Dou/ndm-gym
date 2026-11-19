"""Training loop for humanoid motion tracking on MuJoCo-Warp: a Unitree humanoid (``--robot``: the H1, or the G1 for
reference; robots.humanoid) follows LAFAN1 retargeted to it (motion.sources.lafan1), one clip or many with one policy.
The quadruped trainer's loop (ndm_gym.train_quadruped: its schedule, options, systems options, log and checkpoints) with the
humanoid's scene, task (tasks.humanoid_tracking) and agents (algos.humanoid_algos); the simulator is the quadruped task's
(sim.rigid.quadruped_sim) on the humanoid's scene. The log's error columns are the pelvis' (root_mm) and the root-relative
MPJPE (mpjpe_l_mm) of the collected steps.

Run: ``python -m ndm_gym.train_humanoid --robot h1 --clips walk1_subject1 --log_dir logs/<run>`` (``--clips``: clip names
or globs, comma-separated; default every clip); ``--config list`` for the named configurations.
"""

from __future__ import annotations

import argparse
import fnmatch

NCONMAX, NJMAX = 64, 320  # MuJoCo-Warp's contacts and constraint rows per world
# ndm_gym.train's options that train_humanoid implements beyond the quadruped trainer's: LAFAN1 training's collection steps
# per update, the world model's learning rate and the G1 SuperTrack's balance weight (with --st_terms g1)
SUPPORTED = ("utd", "wm_lr", "st_balance_w")
HUMANOID_DEFAULTS = {"utd": 1.0}  # one collection step per iteration, as the quadruped trainer


def add_humanoid_args(parser: argparse.ArgumentParser):
    """The humanoid's robot, servo and start options."""
    parser.add_argument("--robot", choices=("h1", "g1"), default="h1", help="the humanoid (robots.humanoid.ROBOTS)")
    parser.add_argument("--kp_scale", type=float, default=1.0, help="the servos' stiffness, times the robot's (robots.humanoid)")
    parser.add_argument("--kd_scale", type=float, default=1.0, help="the servos' damping, times the robot's")
    parser.add_argument("--ankle_effort_scale", type=float, default=1.0,
                        help="the ankles' effort limits (the actuators' force limits and the joints' actuator torque limits, and so their "
                        "action scale), times the robot's (robots.humanoid.with_ankle_effort; 1: the robot's, the H1's 40 N m from its URDF); "
                        "a departure for diagnosis")  # fmt: skip
    parser.add_argument("--target_range", choices=("joint", "effort", "free"), default="joint",
                        help="the position targets the servos accept, in the simulator, the world model's imagined steps and the evaluation "
                        "(robots.humanoid.TARGET_RANGES): the joint's range (the default; at a joint's stop no proportional torque toward "
                        "it), that range widened by effort / stiffness on each side (the full effort toward either side at any angle in "
                        "the range), or unlimited (the library G1's); the joints' stops and effort limits stay the robot's")  # fmt: skip
    parser.add_argument("--gains", choices=("robot", "tracking"), default="robot",
                        help="the servos' stiffness and damping: the robot's own (the default; the H1's from unitree_rl_gym) or, the H1 "
                        "only, the stiffer and more damped gains of H1 motion-tracking projects (HumanoidVerse's and ProtoMotions': "
                        "hips 200 / 5, knees 300 / 6, ankles 40 / 2, torso 300 / 6, shoulders and elbows 100 / 2; "
                        "robots.humanoid.with_gains), a departure; the action scale follows the stiffness in use")  # fmt: skip
    parser.add_argument("--ref_ground", choices=("none", "clip", "stance"), default="none",
                        help="a height correction of the reference against the floor when it is built (tasks.humanoid_tracking.ground_"
                        "reference; the files are unchanged): none (the default, the retarget's heights), clip (one vertical offset per "
                        "clip, from its supporting feet's lowest collision points), or stance (clip's offset and a smooth time-varying "
                        "lift of the pelvis where a supporting foot still goes into the floor), a departure from the retarget")  # fmt: skip
    parser.add_argument("--foot", choices=("mesh", "menagerie"), default="mesh",
                        help="the H1's foot collision geoms: the URDF's ankle mesh (its convex hull, a sole 3 cm wide; the default) or "
                        "MuJoCo Menagerie's unitree_h1 capsules (toe 8.8 cm wide; robots.humanoid.load_scene), a departure from the URDF")  # fmt: skip
    parser.add_argument("--term_pos", type=float, default=float("inf"),
                        help="the episode also terminates when the pelvis is this far from the reference's (m; tasks.humanoid_tracking; "
                        "default: no such term, the G1's termination)")  # fmt: skip
    parser.add_argument("--max_episode_steps", type=int, default=0,
                        help="a training episode also ends after this many control steps (the G1's 500; 0: at its clip's end)")
    parser.add_argument("--bin_starts", action="store_true",
                        help="episodes start in 1 s bins drawn by the G1's return-based sampling (sim.rigid.humanoid_sim)")
    parser.add_argument("--fail_reset", type=float, default=0.0,
                        help="with --bin_starts: this share of the starts from the bins before recent failure points (the G1's)")
    parser.add_argument("--reset_noise", choices=("gaussian", "g1"), default="gaussian",
                        help="the start perturbation in training: the quadruped task's Gaussian one (--joint_noise) or the G1's uniform one "
                        "(--joint_noise then perturbs the evaluation's starts only)")
    parser.add_argument("--goal_anchor", choices=("robot", "reference"), default="robot",
                        help="the goal's pelvis offsets from the pelvis, or horizontally from the reference's pelvis (no drift feedback, "
                        "as the G1's goal; tasks.humanoid_tracking.HumanoidTask.goal)")  # fmt: skip
    parser.add_argument("--action_bound", type=float, default=1.0,
                        help="the policy's actions in [-b, b] (b tanh(y / b)): position-target offsets of up to b times the robot's action "
                        "scale, 0.25 effort / stiffness (1: as the quadruped task; the G1's own policy is unbounded)")  # fmt: skip
    parser.add_argument("--st_terms", choices=("task", "g1"), default="task",
                        help="supertrack: the task's tracked quantities (algos.humanoid_algos.HumanoidSuperTrack.TERMS) or the library "
                        "G1's SuperTrack terms (G1_TERMS, calibrated; --st_balance_w on the heights and the up vector)")  # fmt: skip
    parser.add_argument("--accel_mean_velocity", action="store_true",
                        help="supertrack: the world model's position advances with the mean velocity of the control step's physics steps "
                        "(worldmodels.mjcf.mean_velocity_lead), not its last")  # fmt: skip


def parse_args(argv: list[str] | None = None):
    from ndm_gym.train_quadruped import parse_args as parse

    args = parse(argv, "train_humanoid", add_humanoid_args, "the clips: names or globs of the robot's LAFAN1 clips (comma-separated; default: every clip)",
                 SUPPORTED, HUMANOID_DEFAULTS)
    if args.algo == "supermimic" and (args.st_terms != "task" or args.st_balance_w != 1.0 or args.accel_mean_velocity):
        raise SystemExit("train_humanoid: --st_terms, --st_balance_w and --accel_mean_velocity are SuperTrack's options")
    if args.st_balance_w != 1.0 and args.st_terms != "g1":
        raise SystemExit("train_humanoid: --st_balance_w weighs the G1's balance terms (--st_terms g1)")
    if args.fail_reset and not args.bin_starts:
        raise SystemExit("train_humanoid: --fail_reset needs --bin_starts")
    if args.foot != "mesh" and args.robot != "h1":
        raise SystemExit("train_humanoid: --foot is the H1's option")
    if args.gains != "robot" and args.robot != "h1":
        raise SystemExit("train_humanoid: --gains tracking is the H1's option")
    if args.ankle_effort_scale <= 0:
        raise SystemExit("train_humanoid: --ankle_effort_scale must be positive")
    if args.utd <= 0:
        raise SystemExit("train_humanoid: --utd must be positive")
    if args.action_bound <= 0 or args.term_pos <= 0 or args.kp_scale <= 0 or args.kd_scale <= 0:
        raise SystemExit("train_humanoid: --action_bound, --term_pos, --kp_scale and --kd_scale must be positive")
    if args.max_episode_steps < 0 or not 0 <= args.fail_reset <= 1:
        raise SystemExit("train_humanoid: --max_episode_steps is 0 (to the clip's end) or more, --fail_reset a share in [0, 1]")
    return args


def select_clips(clips: list, names: str | list[str] | None) -> list:
    """The clips whose names match ``names`` (names or globs, comma-separated or a list; None: every clip), in their order."""
    if names is None:
        return clips
    pats = [n for n in (names.split(",") if isinstance(names, str) else names) if n]
    for p in pats:
        if not any(fnmatch.fnmatchcase(c.name, p) for c in clips):
            raise ValueError(f"no clip matches {p!r}")
    return [c for c in clips if any(fnmatch.fnmatchcase(c.name, p) for p in pats)]


def build(robot: str, clips: str | list[str] | None, device: str, reward_std_scale: float = 1.0, kp_scale: float = 1.0, kd_scale: float = 1.0,
          term_pos: float = float("inf"), goal_anchor: str = "robot", foot: str = "mesh",
          ankle_effort_scale: float = 1.0, target_range: str = "joint", gains: str = "robot", ref_ground: str = "none",
          score_ground: str | None = None):
    """The scene, kinematics, task and clip names of ``robot``'s clips (``select_clips``); ``foot``: the H1's foot
    collision geoms, ``target_range``: the servos' accepted position targets (robots.humanoid.load_scene; the task's
    setpoint clamps to them); ``ankle_effort_scale``: robots.humanoid.with_ankle_effort's; ``gains``: the servos' gains
    (robots.humanoid.with_gains); ``ref_ground``: the reference's height correction (tasks.humanoid_tracking.ground_reference),
    the reference the policy follows (its goal, its targets' feed-forward pose and its starts) and, unless ``score_ground``
    gives another (the evaluation's), the one its reward, errors and termination measure against."""
    from ndm_gym.motion.sources import lafan1
    from ndm_gym.robots.humanoid import ROBOTS, HumanoidKinematics, load_scene, with_ankle_effort, with_gains
    from ndm_gym.tasks.humanoid_tracking import HumanoidTask, RewardCfg, build_reference

    spec = with_gains(with_ankle_effort(ROBOTS[robot], ankle_effort_scale), gains)
    model = load_scene(spec, kp_scale=kp_scale, kd_scale=kd_scale, foot=foot, target_range=target_range)
    kin = HumanoidKinematics(model, spec, device)
    loaded = select_clips(lafan1.load_dir(spec.motions, spec.num_joints), clips)
    ref = build_reference(loaded, kin, device, ground=ref_ground)
    score = None
    if score_ground is not None and score_ground != ref_ground:
        score = build_reference(loaded, kin, device, ground=score_ground)
    task = HumanoidTask(kin, ref, RewardCfg(term_pos=term_pos).scaled(reward_std_scale), goal_anchor, score_ref=score)
    return model, kin, task, [c.name for c in loaded]


def make_sim(model, task, worlds: int, device: str, joint_noise: float = 0.05, args=None):
    """The humanoid's simulator (sim.rigid.humanoid_sim); ``args``: training's episode options (none: evaluation's
    simulator, QuadrupedSim's episodes)."""
    from ndm_gym.sim.rigid.humanoid_sim import HumanoidSim

    opts = {}
    if args is not None:
        opts = {"max_episode_steps": args.max_episode_steps or None, "bin_starts": args.bin_starts, "fail_reset": args.fail_reset,
                "reset_noise": args.reset_noise, "gamma": args.gamma}  # fmt: skip
    return HumanoidSim(model, task, worlds, device, joint_noise=joint_noise, nconmax=NCONMAX, njmax=NJMAX, **opts)


def main():
    args = parse_args()
    from ndm_gym.algos.humanoid_algos import HUMANOID_ALGOS
    from ndm_gym.train_quadruped import train

    def build_task(dev: str):
        return build(args.robot, args.clips, dev, args.reward_std_scale, args.kp_scale, args.kd_scale, args.term_pos, args.goal_anchor,
                     args.foot, args.ankle_effort_scale, args.target_range, args.gains, args.ref_ground)

    train(args, build_task, HUMANOID_ALGOS, lambda model, task, n, dev: make_sim(model, task, n, dev, args.joint_noise, args), utd=args.utd,
          wm_lr=args.wm_lr)


if __name__ == "__main__":
    main()

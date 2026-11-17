"""Named run configurations (ndm_gym.configs): each one parses with its trainer's own parser, the command line's options come
after it (so they take precedence), the run records the name, and an unknown name or a second --config is an error."""

import sys

import pytest

from ndm_gym import configs


def parse(trainer, argv):
    if trainer == "train":
        from ndm_gym import train

        old = sys.argv
        sys.argv = ["ndm-train", *argv]
        try:
            return train.parse_args()
        finally:
            sys.argv = old
    mod = __import__(f"ndm_gym.{trainer}", fromlist=["parse_args"])
    return mod.parse_args(argv)


EXTRA = {"train": [], "train_hand": ["--clip", "clip"], "train_muscles": []}


@pytest.mark.parametrize("trainer, name", [(t, n) for t, names in configs.CONFIGS.items() for n in names])
def test_every_configuration_parses(trainer, name):
    extra = EXTRA[trainer] + (["--motion_dir", "clip.npz"] if name.startswith("two-g1") else [])
    args = parse(trainer, ["--config", name, *extra])
    assert args.config == name
    algo = "supertrack" if "supertrack" in name else "supermimic"
    assert args.algo == algo


def test_the_command_line_takes_precedence_and_errors():
    argv = configs.expand(["--config", "lafan1-supertrack-fast", "--wm_lr", "1e-3"], "train")
    assert argv.index("--wm_lr", argv.index("--st_quad_w")) > 0 and argv[-4:] == ["--wm_lr", "1e-3", "--config", "lafan1-supertrack-fast"]
    assert parse("train", ["--config", "lafan1-supertrack-fast", "--wm_lr", "1e-3"]).wm_lr == 1e-3
    with pytest.raises(ValueError, match="not a configuration"):
        configs.expand(["--config", "lafan1-supertrack-fastest"], "train")
    with pytest.raises(ValueError, match="not a configuration"):  # each trainer reads its own
        configs.expand(["--config", "hand-supertrack-fast"], "train")
    with pytest.raises(ValueError, match="one configuration"):
        configs.expand(["--config", "lafan1-supertrack-fast", "--config=lafan1-supermimic-fast"], "train")
    assert configs.expand(["--algo", "supertrack"], "train") == ["--algo", "supertrack"]  # no --config: unchanged


# the arguments of the muscles' 3 h RigidFormer runs, seed 1's without the seed (benchmarks/musclevae/results.md)
MUSCLE_RF_RUNS = {
    "muscles-supertrack-rf": (
        "--algo supertrack --st_fail_w 5 --st_balance_w 4 --pi_ema 0.995 --lr 4e-4 --wm_batches 5 --pi_opt muon --wm_only_iters 0 "
        "--pi_batch 2048 --pi_lr_mult 4 --lr_decay_len 300 --lr_decay_by 1400 --wm_backbone rigidformer --rf_muscle_tokens body --rf_dim 128 "
        "--rf_layers 2 --rf_heads 4 --rf_registers 9 --rf_fused_train --rf_fused_features --wm_lr 5e-3 --wm_opt muon --no_sync --fused_adam "
        "--update_graphs --graph_early --collect_graph --reset_masked --ckpt_light --max_hours 3 --iterations 1000000 --ckpt_every 2000 "
        "--last_every_min 10"
    ),
    "muscles-supermimic-rf": (
        "--algo supermimic --reward_std_scale 0.5 --action_noise 0.3 --imag_start_perturb 0.3 --pi_ema 0.995 --lr 4e-4 --wm_batches 8 "
        "--pi_opt muon --wm_only_iters 0 --pi_batch 2048 --pi_lr_mult 2 --lr_decay_len 300 --lr_decay_by 1400 --wm_backbone rigidformer "
        "--rf_muscle_tokens body --rf_dim 128 --rf_layers 2 --rf_heads 4 --rf_registers 9 --rf_fused_train --rf_fused_features --wm_lr 5e-3 "
        "--no_sync --fused_adam --update_graphs --graph_early --collect_graph --reset_masked --ckpt_light --max_hours 3 --iterations 1000000 "
        "--ckpt_every 2000 --last_every_min 10"
    ),
}


def test_the_muscle_rigidformer_configurations_are_their_3h_runs():
    """muscles-supertrack-rf and muscles-supermimic-rf parse to the arguments of the muscles' 3 h RigidFormer runs."""
    for name, run in MUSCLE_RF_RUNS.items():
        got, want = vars(parse("train_muscles", ["--config", name])), vars(parse("train_muscles", run.split()))
        assert got.pop("config") == name and want.pop("config") is None
        assert got == want, name

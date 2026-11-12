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

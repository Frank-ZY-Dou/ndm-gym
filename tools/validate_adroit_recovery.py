"""Full-capacity fresh-process recovery for the shared Adroit trainer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_door_full_recovery import main
from ndm_gym.tasks.adroit import training_arguments

if __name__ == '__main__':
    cfg, remaining = training_arguments()
    main(remaining, training_module='ndm_gym.train_adroit', training_args=('--nail-tracking-std', str(cfg.nail_std)))

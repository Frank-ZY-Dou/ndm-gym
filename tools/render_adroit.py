"""Native suite previews using the shared NDM-marked Adroit renderer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_adroit_door import main
from ndm_gym.motion.adroit import AdroitReference

if __name__ == '__main__':
    main(reference_cls=AdroitReference, native_camera=True)

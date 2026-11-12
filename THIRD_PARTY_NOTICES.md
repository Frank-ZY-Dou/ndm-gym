# Third-party notices

NDM Gym started from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (MIT, ETH Zurich, Chong Zhang); its
copyright line is in [`LICENSE`](LICENSE).

## Submodules

The git submodule `third_party/supermimic` ([zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) @
`ee0608f`, MIT, Chong Zhang), used unmodified for the equivalence tests, keeps its own license file and copyright notices.

## RoboKit

`tests/test_robokit_rec.py` and `tests/test_robokit_ndm.py` adapt the cart model and planner settings of the SimMPC guide of
[RoboKit](https://github.com/openrobokit/robokit) (MIT, Copyright (c) 2025 OpenRoboKit). This library's RoboKit modules
(`ndm_gym/sim/robokit_*.py`, `ndm_gym/wmbench/*robokit*.py`) patch RoboKit's classes at run time and copy none of its files.

## Ported code

The options of `--algo blinkmimic` ([`docs/blinkmimic.md`](docs/blinkmimic.md)) are ported from
[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) @ `d264658`, which is derived from
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack), both under the MIT License reproduced below. The ported
files carry BlinkMimic's notice: `ndm_gym/algos/blinkmimic.py`, `ndm_gym/algos/eprobe.py`, `ndm_gym/core/muon.py`,
`ndm_gym/sim/rigid/reset_stats.py`, `ndm_gym/tasks/contact.py` and `ndm_gym/worldmodels/semi_implicit.py`. Adapted parts of
`ndm_gym/train.py`, `evaluate.py`, `buffer.py`, `algos/common.py`, `core/normalizer.py`, `sim/rigid/isaaclab_mjwarp.py`,
`tests/test_eprobe.py` and `tests/test_blinkmimic.py` are under the same license.

## BlinkMimic

```
MIT License

Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

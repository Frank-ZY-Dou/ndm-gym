"""NablaMimic's configs/ours.args with one change, --no-remat removed: the windows are then rematerialized in the backward pass
(jax.checkpoint with the engine's remat policy, its train.py:177), which takes less GPU memory and more time per update. The same
algorithm and hyperparameters; the gradients equal those without rematerialization up to rounding (NablaEngine, docs/api.md, Remat).
A configuration file of NablaMimic cannot remove an included option, so the file is written out in full, as its own
configs/reported/ablations/no_pods.args is. Nothing is copied into this repository: the options are read from the checkout with its
own reader and checked with its own parser. Run by run.sh in NablaMimic's environment, on the CPU:

  python -I remat_config.py <output file>
"""

import subprocess
import sys
from pathlib import Path

from nablamimic import train
from nablamimic.params import REPO, read_params


def main(out):
    src = REPO / "configs" / "ours.args"
    args = read_params(src)
    if args.count("--no-remat") != 1:
        raise SystemExit(f"{src} does not hold --no-remat exactly once: {args}")
    args.remove("--no-remat")
    lines, cur = [], []
    for x in args:  # one option per line
        if x.startswith("--") and cur:
            lines.append(" ".join(cur))
            cur = []
        cur.append(x)
    lines.append(" ".join(cur))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    head = (f"# NablaMimic's configs/ours.args at {commit}, resolved by its own reader, without --no-remat (the only change):\n"
            "# the windows are rematerialized in the backward pass. Not a published configuration.\n")  # fmt: skip
    Path(out).write_text(head + "\n".join(lines) + "\n")
    parse = lambda a: vars(train.build_parser().parse_args(list(a) + ["--out", "unused"]))  # noqa: E731
    new, published = parse(read_params(out)), parse(read_params(src))
    diff = {k: (published[k], new[k]) for k in new if new[k] != published[k]}
    if diff != {"no_remat": (True, False)}:
        raise SystemExit(f"{out} differs from {src} in {diff}, not in no_remat alone")
    print(f"{out}: {len(lines)} options of {src} at {commit}; the parsed arguments differ in no_remat alone (True -> False)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])

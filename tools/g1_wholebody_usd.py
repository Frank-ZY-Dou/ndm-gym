"""tools/build_g1_wholebody_clip.py's MuJoCo models as USD for the whole-body simulator, with the standalone Isaac Sim
MJCF importer (as tools/mjcf_to_usd.py converts the G1).

    python tools/g1_wholebody_usd.py <build_g1_wholebody_clip.py's out dir> data/hdmi/g1_inspire/<name>

Writes <usd dir>/<model>/<model>.usda for robot.xml, object.xml and table.xml. The object and the table are single
rigid bodies (Isaac Lab's RigidObject), so their articulation roots, PhysX's and Newton's, are removed.
"""

import os
import sys

from pxr import Plug, Sdf

import isaacsim.asset.importer.utils.impl as importer_utils

Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(importer_utils.__file__), "schemas"))
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig  # noqa: E402

ARTICULATION_ROOTS = ("PhysicsArticulationRootAPI", "NewtonArticulationRootAPI")


def remove_articulation_roots(directory: str) -> int:
    """Removes the articulation root schemas from every prim of every USD layer under ``directory``; returns how many."""
    removed = 0
    for root, _, files in os.walk(directory):
        for name in files:
            if not name.endswith((".usda", ".usd", ".usdc")):
                continue
            layer = Sdf.Layer.FindOrOpen(os.path.join(root, name))
            specs = []
            layer.Traverse(Sdf.Path.absoluteRootPath, lambda path: specs.append(layer.GetPrimAtPath(path)) if path.IsPrimPath() else None)
            changed = False
            for spec in specs:
                ops = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else None
                if ops is None:
                    continue
                items = {k: list(getattr(ops, k)) for k in ("explicitItems", "prependedItems", "appendedItems")}
                if any(t in ARTICULATION_ROOTS for v in items.values() for t in v):
                    new = Sdf.TokenListOp()
                    if ops.isExplicit:
                        new.explicitItems = [t for t in items["explicitItems"] if t not in ARTICULATION_ROOTS]
                    else:
                        new.prependedItems = [t for t in items["prependedItems"] if t not in ARTICULATION_ROOTS]
                        new.appendedItems = [t for t in items["appendedItems"] if t not in ARTICULATION_ROOTS]
                    spec.SetInfo("apiSchemas", new)
                    removed, changed = removed + 1, True
            if changed:
                layer.Save()
    return removed


source, out = (os.path.abspath(p) for p in sys.argv[1:3])
for model in ("robot", "object", "table"):
    config = MJCFImporterConfig(mjcf_path=os.path.join(source, f"{model}.xml"), usd_path=out)
    print(model, MJCFImporter(config).import_mjcf())
    if model != "robot":
        print(f"  {remove_articulation_roots(os.path.join(out, model))} articulation root(s) removed")

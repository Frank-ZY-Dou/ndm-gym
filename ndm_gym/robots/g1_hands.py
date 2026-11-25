"""A Unitree G1 with a SPIDER hand-object clip's hands on its wrists (tools/build_g1_hand_task.py).

In a SPIDER clip each hand's base floats on three slides and three hinges. Here the bases are fixed to the G1's wrists in
place of its rubber hands, the fingers along the forearm and the palms toward the body's middle. The G1 stands at a
small table whose top is the clip's floor under the resting object, and its arms are solved by inverse kinematics
(``ArmSolver``) so that each base keeps the clip's pose, clear of the rest of the robot. The fingers and the object move
as in the clip.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

SIDES = ("right", "left")
FINGERS = ("index", "middle", "ring", "pinky")
OBJECT = "right_object"  # SPIDER's object body
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
STANCE = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363}  # the G1's standing pose
ARM_REST = {"shoulder_pitch": 0.2, "shoulder_roll": 0.2, "elbow": 0.6}  # the left arm's; the right arm's roll is mirrored
ROTATION_WEIGHT = 0.15  # a base's rotation error (rad) counted as this many meters
POSTURE, SMOOTH = 1e-5, 1e-4  # weights of the distance from the rest pose and from the previous frame
WAIST_WEIGHT = 10.0  # the waist's share of the posture term, against 1 for an arm's joint
LIMIT_MARGIN, LIMIT_WEIGHT = 0.15, 1e-3  # the placement's penalty on joints closer than the margin (rad) to a limit
CLEARANCE, CLEARANCE_WEIGHT = 0.01, 0.5  # the arms' least distance from the rest of the robot (m), its weight against a base's position
PLACEMENT_TRACKING, PLACEMENT_POSTURE = 1e3, 2e-5  # the placement's weights of the bases' errors and of the distance from the rest pose
TABLE_THICKNESS, TABLE_MARGIN = 0.03, 0.02  # the table top's thickness and its margin around the resting object (m)


def absolute_assets(spec: mujoco.MjSpec, directory: str) -> None:
    """Mesh and texture files as absolute paths, so that bodies of one model can be attached to another."""
    directory = os.path.abspath(directory)
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = os.path.normpath(os.path.join(directory, spec.meshdir, mesh.file))
    for texture in spec.textures:
        if texture.file:
            texture.file = os.path.normpath(os.path.join(directory, spec.texturedir, texture.file))
    spec.meshdir = spec.texturedir = ""


def yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def quat_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def pelvis_quat(yaw: float, tilt=None) -> np.ndarray:
    """The pelvis' orientation: ``tilt`` (a quaternion without yaw; None: upright) turned by ``yaw`` about the vertical."""
    quat = np.array(yaw_quat(yaw))
    if tilt is not None:
        mujoco.mju_mulQuat(quat, np.array(yaw_quat(yaw)), np.asarray(tilt, dtype=np.float64))
    return quat


def hand_axes(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, float]:
    """The hand's distal, radial and palmar directions as the columns of a matrix in its base's frame (toward the fingers,
    toward the index's side, out of the palm), and how far the base's meshes reach behind its origin (m)."""
    data = mujoco.MjData(model)
    base = model.body(f"{side}_hand_base").id
    tips = [model.site(f"track_hand_{side}_{finger}_tip").id for finger in FINGERS]

    def in_base(points: np.ndarray) -> np.ndarray:
        return (points - data.xpos[base]) @ data.xmat[base].reshape(3, 3)

    mujoco.mj_kinematics(model, data)
    knuckles = in_base(np.stack([data.xpos[model.body(f"{side}_{finger}_proximal").id] for finger in FINGERS]))
    open_tips = in_base(data.site_xpos[tips])
    behind = 0.0
    for geom in np.flatnonzero((model.geom_bodyid == base) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
        placed = in_base(mesh_points(model, data, geom))
        behind = min(behind, float((placed @ (knuckles.mean(0) / np.linalg.norm(knuckles.mean(0)))).min()))
    for finger in FINGERS:
        data.qpos[model.joint(f"{side}_{finger}_proximal_joint").qposadr[0]] = 1.0
    mujoco.mj_kinematics(model, data)
    curl = (in_base(data.site_xpos[tips]) - open_tips).mean(0)

    distal = knuckles.mean(0) / np.linalg.norm(knuckles.mean(0))
    radial = knuckles[0] - knuckles[-1]
    radial -= (radial @ distal) * distal
    radial /= np.linalg.norm(radial)
    palmar = np.cross(distal, radial)
    palmar *= np.sign(palmar @ curl)
    return np.stack((distal, radial, palmar), axis=1), -behind


def mesh_points(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """A mesh geom's vertices in the world frame."""
    mesh = model.geom_dataid[geom]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    return data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T


def mount_rotation(axes: np.ndarray, side: str) -> np.ndarray:
    """The hand's base in the wrist's frame: the fingers along the forearm (the wrist's x), the index's side up (its z) and
    the palm toward the body's middle, as a person's hand with the forearm held forward."""
    inward = -1.0 if side == "left" else 1.0
    wrist = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, inward], [0.0, 1.0, 0.0]])
    rotation = wrist @ axes.T
    if np.linalg.det(rotation) < 0.999:
        raise ValueError(f"the {side} hand's axes are not those of a {side} hand")
    return rotation


def hand_mounts(hand_model: mujoco.MjModel, g1_model: mujoco.MjModel, sides: list[str]) -> dict:
    """Per side where the hand's base sits in the wrist's frame: in place of the rubber hand, turned as mount_rotation."""
    mounts = {}
    for side in sides:
        axes, behind = hand_axes(hand_model, side)
        mounts[side] = (g1_model.body_pos[g1_model.body(f"{side}_rubber_hand").id] + np.array([behind, 0.0, 0.0]), mount_rotation(axes, side))
    return mounts


def subtree(scene_xml: str, body: str, joints: tuple[str, ...] = ()) -> mujoco.MjSpec:
    """The clip's scene reduced to one of its bodies, without ``joints``, with the meshes and materials that body uses."""
    spec = mujoco.MjSpec.from_file(scene_xml)
    absolute_assets(spec, os.path.dirname(scene_xml))
    for elements in (spec.actuators, spec.sensors, spec.tendons, spec.equalities, spec.pairs, spec.excludes, spec.cameras, spec.lights):
        for element in list(elements):
            spec.delete(element)
    for joint in joints:
        spec.delete(spec.joint(joint))
    for other in list(spec.worldbody.bodies):
        if other.name != body:
            spec.delete(other)
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    meshes, materials = {geom.meshname for geom in spec.geoms}, {geom.material for geom in spec.geoms}
    for mesh in list(spec.meshes):
        if mesh.name not in meshes:
            spec.delete(mesh)
    for material in list(spec.materials):
        if material.name not in materials:
            spec.delete(material)
    if not spec.materials:
        for texture in list(spec.textures):
            spec.delete(texture)
    return spec


def attach(parent: mujoco.MjSpec, frame, child: mujoco.MjSpec, body: str) -> None:
    """``body`` of ``child`` on ``frame`` of ``parent``. Its bodies, joints, geoms and sites keep their names; the assets and
    defaults copied with it are named after the body."""
    prefix = body + "/"
    frame.attach_body(child.body(body), prefix, "")
    for elements in (parent.bodies, parent.joints, parent.geoms, parent.sites):
        for element in elements:
            if element.name.startswith(prefix):
                element.name = element.name[len(prefix) :]


def build(g1_xml: str, scene_xml: str, sides: list[str], mounts: dict, pelvis: np.ndarray, table: dict | None = None, tilt=None,
          floating: bool = False):  # fmt: skip
    """The G1 with the clip's hands on its wrists and the clip's object, its pelvis fixed at ``pelvis`` (x, y, z, yaw) with
    ``tilt`` (pelvis_quat's), or with ``floating`` on its free joint. With ``table`` a floor at height ``table["floor"]`` and a
    small table whose top is at ``table["top"]`` (default 0), turned by ``table["yaw"]`` (default the pelvis'). Returns the
    model's spec and the specs attached to it (a scene's assets can be attached only once)."""
    g1 = mujoco.MjSpec.from_file(g1_xml)
    absolute_assets(g1, os.path.dirname(g1_xml))
    if not floating:
        g1.delete(g1.joint("floating_base_joint"))
        g1.body("pelvis").pos, g1.body("pelvis").quat = pelvis[:3], pelvis_quat(pelvis[3], tilt)
    for geom in g1.geoms:  # the collision shapes in the group the hands' scenes keep theirs in
        if geom.contype or geom.conaffinity:
            geom.group = 3

    attached = []
    for side in sides:
        g1.delete(g1.body(f"{side}_rubber_hand"))
        hand = subtree(scene_xml, f"{side}_hand_base", tuple(f"{side}_{kind}_{axis}" for kind in ("pos", "rot") for axis in "xyz"))
        position, rotation = mounts[side]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.flatten())
        attach(g1, g1.body(f"{side}_wrist_yaw_link").add_frame(pos=position, quat=quat), hand, f"{side}_hand_base")
        attached.append(hand)
    attached.append(subtree(scene_xml, OBJECT))
    attach(g1, g1.worldbody.add_frame(), attached[-1], OBJECT)

    if table is not None:
        top, floor, wood = table.get("top", 0.0), table["floor"], (0.80, 0.69, 0.55, 1)
        g1.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05), pos=(0, 0, floor), group=2)
        stand = g1.worldbody.add_body(name="table", pos=(*table["center"], top), quat=yaw_quat(table.get("yaw", pelvis[3])))
        height = top - floor - TABLE_THICKNESS
        stand.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=(*table["half_size"], TABLE_THICKNESS / 2),
                       pos=(0, 0, -TABLE_THICKNESS / 2), rgba=wood, group=2)  # fmt: skip
        stand.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.03, 0.03, height / 2), pos=(0, 0, -TABLE_THICKNESS - height / 2), rgba=wood, group=2)
        stand.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.14, 0.14, 0.01), pos=(0, 0, floor - top + 0.01), rgba=wood, group=2)
    return g1, attached


class ArmSolver:
    """Inverse kinematics of the G1's arms, with ``waist`` of its waist too, for the poses of the hands' bases, the arms kept
    clear of the rest of the robot (Levenberg-Marquardt, joint limits by clipping). ``posture`` holds the other joints (default:
    the legs' standing pose). Without a target the arms stay at ``rest``, moved only as far as their clearance needs; with
    ``free_hands`` (the hands hold nothing) the arms and hands also stay clear of the object, the hands of the table."""

    def __init__(self, model: mujoco.MjModel, sides: list[str], waist: bool, posture: dict[str, float] | None = None):
        self.model, self.data = model, mujoco.MjData(model)
        names = self.names = (list(WAIST) if waist else []) + [f"{side}_{joint}_joint" for side in sides for joint in ARM]
        self.qpos = np.array([model.joint(name).qposadr[0] for name in names])
        self.dofs = np.array([model.joint(name).dofadr[0] for name in names])
        self.low, self.high = model.jnt_range[[model.joint(name).id for name in names]].T
        self.bases = [model.body(f"{side}_hand_base").id for side in sides]
        # collision shapes: each arm's links from its upper arm on, each hand's, the rest of the robot's, the table's, the object's
        link = lambda geom: model.body(model.geom_bodyid[geom]).name  # noqa: E731
        solid = [geom for geom in range(model.ngeom) if model.geom_bodyid[geom] and (model.geom_contype[geom] or model.geom_conaffinity[geom])]
        arm_parts, hand_parts = ("shoulder", "elbow", "wrist"), ("hand_base", "thumb", *FINGERS)
        self.links = [[geom for geom in solid if link(geom).startswith(side) and any(part in link(geom) for part in ("shoulder_yaw", "elbow", "wrist"))]
                      for side in sides]  # fmt: skip
        self.hands = [[geom for geom in range(model.ngeom) if model.geom_group[geom] == 3 and link(geom).startswith(side)
                       and any(part in link(geom) for part in hand_parts)] for side in sides]  # fmt: skip
        self.body = [geom for geom in solid if link(geom) != "table" and not any(part in link(geom) for part in arm_parts + hand_parts)]
        self.scene = [geom for geom in solid if link(geom) == "table"]
        self.object = [geom for geom in range(model.ngeom) if model.geom_group[geom] == 3 and link(geom) == OBJECT]
        self.free_hands, self.posture = False, POSTURE
        self.segment = np.zeros(6)
        self.rest = np.zeros(len(names))
        for i, name in enumerate(names):
            for joint, value in ARM_REST.items():
                if name.endswith(f"{joint}_joint"):
                    self.rest[i] = -value if joint == "shoulder_roll" and name.startswith("right") else value
        self.weight = np.array([WAIST_WEIGHT if name in WAIST else 1.0 for name in names])
        posture = {f"{side}_{joint}_joint": value for side in SIDES for joint, value in STANCE.items()} if posture is None else posture
        for name, value in posture.items():
            self.data.qpos[model.joint(name).qposadr[0]] = value
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def pairs(self) -> list[tuple[int, list[int]]]:
        """Each shape of the arms and the hands with the shapes it stays clear of: an arm's links of the rest of the robot, the
        table and the other arm's links; a hand of the rest of the robot. With ``free_hands`` both also of the object, and the
        hands of the table."""
        pairs, loose = [], self.object if self.free_hands else []
        for arm, links in enumerate(self.links):
            others = [geom for other, shapes in enumerate(self.links) if other != arm for geom in shapes]
            pairs += [(geom, self.body + self.scene + others + loose) for geom in links]
            pairs += [(geom, self.body + (self.scene + loose if self.free_hands else [])) for geom in self.hands[arm]]
        return pairs

    def linearize(self, q: np.ndarray, target, jacobian: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """The bases' position and weighted rotation errors for ``target`` (the bases' positions and rotations; None: no such
        terms), then the arms' and hands' missing clearance, for the joints ``q`` and, with ``jacobian``, their derivative."""
        self.data.qpos[self.qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        residual, rows = [], []
        quat, rotation_error = np.zeros(4), np.zeros(3)
        for base, position, rotation in zip(self.bases, *target, strict=True) if target is not None else ():
            mujoco.mju_mat2Quat(quat, (rotation @ self.data.xmat[base].reshape(3, 3).T).flatten())
            mujoco.mju_quat2Vel(rotation_error, quat, 1.0)
            residual += [position - self.data.xpos[base], ROTATION_WEIGHT * rotation_error]
            if jacobian:
                mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, base)
                rows += [self.jacp[:, self.dofs].copy(), ROTATION_WEIGHT * self.jacr[:, self.dofs]]
        for geom, obstacles in self.pairs():  # a shape closer than its clearance is pushed away
            nearest, segment, other = CLEARANCE, None, -1
            for obstacle in obstacles:
                distance = mujoco.mj_geomDistance(self.model, self.data, geom, obstacle, CLEARANCE, self.segment)
                if distance < nearest:
                    nearest, segment, other = distance, self.segment.copy(), obstacle
            residual.append(np.array([CLEARANCE_WEIGHT * (CLEARANCE - nearest)]))
            if jacobian:
                row = np.zeros(len(self.dofs))
                if segment is not None and abs(nearest) > 1e-6:
                    away = (segment[:3] - segment[3:]) / nearest  # from the obstacle to the arm
                    mujoco.mj_jac(self.model, self.data, self.jacp, None, segment[:3], self.model.geom_bodyid[geom])
                    row = away @ self.jacp[:, self.dofs]
                    mujoco.mj_jac(self.model, self.data, self.jacp, None, segment[3:], self.model.geom_bodyid[other])
                    row = CLEARANCE_WEIGHT * (row - away @ self.jacp[:, self.dofs])
                rows.append(row[None])
        return np.concatenate(residual), np.concatenate(rows) if jacobian else None

    def cost(self, q: np.ndarray, target, previous: np.ndarray | None) -> float:
        residual, _ = self.linearize(q, target, jacobian=False)
        smooth = SMOOTH * ((q - previous) ** 2).sum() if previous is not None else 0.0
        return float(residual @ residual + self.posture * (self.weight * (q - self.rest) ** 2).sum() + smooth)

    def solve(self, q: np.ndarray, target, previous: np.ndarray | None = None, iterations: int = 60) -> np.ndarray:
        damping, cost = 1e-4, self.cost(q, target, previous)
        for _ in range(iterations):
            residual, jacobian = self.linearize(q, target)
            hessian = jacobian.T @ jacobian + self.posture * np.diag(self.weight)
            gradient = jacobian.T @ residual + self.posture * self.weight * (self.rest - q)
            if previous is not None:
                hessian += SMOOTH * np.eye(len(q))
                gradient += SMOOTH * (previous - q)
            while damping < 1e3:
                candidate = np.clip(q + np.linalg.solve(hessian + damping * np.eye(len(q)), gradient), self.low, self.high)
                candidate_cost = self.cost(candidate, target, previous)
                if candidate_cost < cost:
                    break
                damping *= 5.0
            else:
                return q
            moved, q, cost, damping = np.abs(candidate - q).max(), candidate, candidate_cost, max(damping / 3.0, 1e-7)
            if moved < 1e-7:
                break
        return q

    def start(self, target, starts: int, rng: np.random.Generator) -> np.ndarray:
        """The best of the solutions from the rest pose and from random poses around it."""
        guesses = [self.rest] + [np.clip(self.rest + rng.uniform(-1.0, 1.0, len(self.rest)) * (self.high - self.low) / 4, self.low, self.high)
                                 for _ in range(starts - 1)]  # fmt: skip
        solutions = [self.solve(guess, target) for guess in guesses]
        return min(solutions, key=lambda q: self.cost(q, target, None))

    def track(self, targets: list, rng: np.random.Generator) -> np.ndarray:
        """The arms' joints for every frame's target, each frame's solution starting from the previous one."""
        solutions = [self.start(targets[0], 32, rng)]
        for target in targets[1:]:
            solutions.append(self.solve(solutions[-1], target, previous=solutions[-1]))
        return np.array(solutions)

    def clearance(self, q: np.ndarray, within: float = 0.3) -> float:
        """The smallest distance of the arms and hands from the shapes they stay clear of (m; negative: they overlap), up to
        ``within``."""
        self.data.qpos[self.qpos] = q
        mujoco.mj_kinematics(self.model, self.data)
        return min((mujoco.mj_geomDistance(self.model, self.data, geom, obstacle, within, None)
                    for geom, obstacles in self.pairs() for obstacle in obstacles), default=within)  # fmt: skip

    def errors(self, q: np.ndarray, target) -> np.ndarray:
        """Per base its position error (m) and rotation error (rad)."""
        residual, _ = self.linearize(q, target, jacobian=False)
        parts = residual[: 6 * len(self.bases)].reshape(len(self.bases), 2, 3)
        return np.stack((np.linalg.norm(parts[:, 0], axis=-1), np.linalg.norm(parts[:, 1], axis=-1) / ROTATION_WEIGHT), axis=-1)


def place(solver: ArmSolver, model: mujoco.MjModel, targets: list, guess: np.ndarray, rng: np.random.Generator, full_turn: bool,
          tilt=None) -> np.ndarray:  # fmt: skip
    """The pelvis' position and yaw for which the sampled poses are tracked best: a coarse scan around ``guess`` (x, y, z, yaw),
    with ``full_turn`` over every facing direction, then a pattern search."""
    pelvis = model.body("pelvis").id

    def cost(pose: np.ndarray) -> float:
        model.body_pos[pelvis], model.body_quat[pelvis] = pose[:3], pelvis_quat(pose[3], tilt)
        q, total = solver.start(targets[0], 4, rng), 0.0
        for target in targets:
            q = solver.solve(q, target, iterations=30)
            residual, _ = solver.linearize(q, target, jacobian=False)
            near = np.maximum(LIMIT_MARGIN - np.minimum(q - solver.low, solver.high - q), 0.0)
            total += float(PLACEMENT_TRACKING * residual @ residual + LIMIT_WEIGHT * (near**2).sum() + PLACEMENT_POSTURE * ((q - solver.rest) ** 2).sum())
        return total / len(targets)

    forward = lambda yaw: np.array([np.cos(yaw), np.sin(yaw), 0.0])  # noqa: E731
    candidates = [np.array([*(guess[:3] - back * forward(guess[3] + turn) - np.array([0, 0, down])), guess[3] + turn])
                  for turn in np.radians(np.arange(-180, 180, 30) if full_turn else np.arange(-90, 91, 30))
                  for back in (0.20, 0.28, 0.36) for down in (0.05, 0.15, 0.25)]  # fmt: skip
    costs = [cost(candidate) for candidate in candidates]
    best, best_cost = candidates[int(np.argmin(costs))], min(costs)
    steps = np.array([0.05, 0.05, 0.05, np.radians(15.0)])
    for _ in range(5):
        improved = True
        while improved:
            improved = False
            for axis in range(4):
                for sign in (-1.0, 1.0):
                    candidate = best.copy()
                    candidate[axis] += sign * steps[axis]
                    candidate_cost = cost(candidate)
                    if candidate_cost < best_cost:
                        best, best_cost, improved = candidate, candidate_cost, True
        steps /= 2.0
    return best


def hand_targets(model: mujoco.MjModel, data: mujoco.MjData, clip: np.ndarray, sides: list[str]) -> list:
    """Per frame of the clip the bases' positions and rotations."""
    bases = [model.body(f"{side}_hand_base").id for side in sides]
    targets = []
    for qpos in clip:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        targets.append((data.xpos[bases].copy(), data.xmat[bases].reshape(-1, 3, 3).copy()))
    return targets


def placement_guess(targets: list, sides: list[str]) -> np.ndarray:
    """A first pelvis pose (x, y, z, yaw): at the bases' mean position, facing them with the left hand on its left."""
    mean = np.mean([positions for positions, _ in targets], axis=0)
    across = mean[sides.index("left")] - mean[sides.index("right")] if len(sides) == 2 else None
    yaw = float(np.arctan2(across[1], across[0]) - np.pi / 2) if across is not None else 0.0
    return np.array([*mean.mean(0), yaw])


def placement_frames(targets: list) -> list[int]:
    """The frames the placement is scored on: a dozen over the clip and those where a base is farthest along an axis."""
    positions = np.array([positions for positions, _ in targets])
    extremes = np.concatenate((np.argmin(positions, axis=0).ravel(), np.argmax(positions, axis=0).ravel()))
    return sorted(set(range(0, len(targets), max(1, len(targets) // 12))) | {int(frame) for frame in extremes})


def rest_area(model: mujoco.MjModel, data: mujoco.MjData, clip: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the object rests on the clip's floor: the center and the half sizes of the rectangle around it, with its margin,
    in the frame turned by ``yaw``."""
    body = model.body(OBJECT).id
    height = clip[:, model.jnt_qposadr[model.body_jntadr[body]] + 2]
    turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    points = []
    for qpos in clip[height < height.min() + 0.005]:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        for geom in np.flatnonzero((model.geom_bodyid == body) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
            points.append(mesh_points(model, data, geom)[:, :2] @ turn)
    low, high = np.concatenate(points).min(0), np.concatenate(points).max(0)
    return turn @ ((low + high) / 2), (high - low) / 2 + TABLE_MARGIN


def lift_frame(object_height: np.ndarray) -> int:
    """The first frame with the object off the table."""
    return int(np.argmax(object_height > object_height.min() + 0.01))


def soften(solutions: np.ndarray, until: int, sigma: float = 3.0, fade: int = 6) -> np.ndarray:
    """The arms' joints smoothed over time by a Gaussian of ``sigma`` frames up to frame ``until``, fading to the solved ones
    over the ``fade`` frames before it. Before the hands lift the object the clip's hands jump toward it, faster than the G1's
    joints turn."""
    reach = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma) ** 2)
    padded = np.concatenate((np.repeat(solutions[:1], reach, axis=0), solutions, np.repeat(solutions[-1:], reach, axis=0)))
    smooth = np.stack([np.convolve(padded[:, joint], kernel / kernel.sum(), mode="valid") for joint in range(solutions.shape[1])], axis=1)
    weight = np.clip((until - np.arange(len(solutions))) / fade, 0.0, 1.0)[:, None]
    return weight * smooth + (1 - weight) * solutions


def freeze(spec: mujoco.MjSpec, name: str, body: str, angle: float) -> None:
    """The hinge ``name`` of ``body`` removed, the body turned and moved as the joint at ``angle`` turns it."""
    joint, body = spec.joint(name), spec.body(body)
    axis, anchor = np.asarray(joint.axis, dtype=np.float64), np.asarray(joint.pos, dtype=np.float64)
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, axis / np.linalg.norm(axis), angle)
    turned_anchor = np.zeros(3)
    mujoco.mju_rotVecQuat(turned_anchor, anchor, turn)
    shift = np.zeros(3)
    mujoco.mju_rotVecQuat(shift, anchor - turned_anchor, np.asarray(body.quat, dtype=np.float64))
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(body.quat, dtype=np.float64), turn)
    body.pos, body.quat = np.asarray(body.pos) + shift, quat
    spec.delete(joint)

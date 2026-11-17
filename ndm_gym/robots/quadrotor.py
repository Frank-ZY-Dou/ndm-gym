"""A racing quadrotor as one rigid body: the open-design 5-inch drone of Race Against the Machine (Bosello et al., RA-L 2024).

No MJCF of the drone is published; ``mjcf`` writes one from the paper's specification: 870 g with the battery, 215 mm between
diagonal propellers, a thrust-to-weight ratio of 7.5 measured on a load cell. Not in the paper, and estimates here
(``QuadrotorSpec``): the inertia (2.9, 2.4 and 5.0 g m^2, of the size of a 5-inch racing quadrotor's of this mass: four
motors with their propellers, 45 g each, on the arms and the rest as a box of 14 x 6 x 7 cm give 1.5, 2.4 and 3.4 g m^2),
a linear drag and the rate controller's gains. ``evaluate_quadrotor --inertia_scale`` (and ``--drag_scale``,
``--mass_scale``) flies a policy on a model with other values.

The drone is commanded as its flight controller is, by a collective thrust and body rates (CTBR): the thrust is a force along
the body's z axis; each body rate is tracked by a proportional controller's torque about its axis, limited to what the
propellers' differential thrust gives. Both are MuJoCo actuators on a site (a motor and three velocity servos), so the rate
loop runs at the physics rate inside the simulator. There is no contact: the flights stay in the air. The integrator is
MuJoCo's Euler, which MuJoCo and MuJoCo-Warp step alike here (the drag implicitly, the servos explicitly; with implicitfast
the two treat a site's velocity servo differently and the body rates of one control step differ by several per cent).
"""

from __future__ import annotations

from dataclasses import dataclass

CONTROL_DT = 0.01  # the task's control step (100 Hz); the library's other tasks run at 50 Hz (ndm_gym.state.CONTROL_DT)
NUM_U = 4  # collective thrust (N), body rates about x, y, z (rad/s)
GRAVITY = 9.81


@dataclass(frozen=True)
class QuadrotorSpec:
    mass: float = 0.87  # kg, with the battery (the paper)
    thrust_to_weight: float = 7.5  # (the paper, load cell)
    diagonal: float = 0.215  # m between diagonal propellers (the paper)
    inertia: tuple[float, float, float] = (0.0029, 0.0024, 0.0050)  # kg m^2, an estimate (the module's docstring)
    drag: float = 0.35  # N s/m on the linear velocity, estimate
    rate_gain: tuple[float, float, float] = (0.145, 0.12, 0.10)  # N m s/rad: time constants of 20, 20 and 50 ms
    torque_limit: tuple[float, float, float] = (2.0, 2.0, 0.3)  # N m
    rate_limit: float = 20.0  # rad/s, the largest commanded body rate
    timestep: float = 0.0025  # s, the physics step: four per control step
    # the policy's action in [-1, 1]: offsets about the reference's feedforward thrust and body rates
    thrust_scale: float = 17.0  # N (about 2 g)
    rate_scale: tuple[float, float, float] = (6.0, 6.0, 3.0)  # rad/s

    @property
    def weight(self) -> float:
        return self.mass * GRAVITY

    @property
    def max_thrust(self) -> float:
        return self.thrust_to_weight * self.weight


def mjcf(spec: QuadrotorSpec = QuadrotorSpec()) -> str:
    """The drone's MJCF: a free body, the thrust and the three rate servos on its site, no contacts."""
    ix, iy, iz = spec.inertia
    arm = spec.diagonal / 2 * 0.7071
    servos = "\n".join(
        f'    <velocity name="rate_{n}" site="thrust" gear="{g}" kv="{kv}" ctrlrange="{-spec.rate_limit} {spec.rate_limit}" '
        f'forcerange="{-lim} {lim}"/>'
        for n, g, kv, lim in zip("xyz", ("0 0 0 1 0 0", "0 0 0 0 1 0", "0 0 0 0 0 1"), spec.rate_gain, spec.torque_limit, strict=True)
    )
    arms = "\n".join(
        f'      <geom type="capsule" fromto="0 0 0 {sx * arm} {sy * arm} 0" size="0.008" rgba="0.15 0.15 0.17 1"/>\n'
        f'      <geom type="cylinder" pos="{sx * arm} {sy * arm} 0.012" size="0.0635 0.003" rgba="0.2 0.5 0.9 0.5"/>'
        for sx in (1, -1) for sy in (1, -1)
    )  # fmt: skip
    return f"""<mujoco model="racing_quadrotor">
  <option timestep="{spec.timestep}" integrator="Euler" gravity="0 0 {-GRAVITY}">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0" density="0"/>
  </default>
  <worldbody>
    <body name="drone" pos="0 0 1">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{spec.mass}" diaginertia="{ix} {iy} {iz}"/>
      <geom type="box" size="0.045 0.025 0.02" rgba="0.15 0.15 0.17 1"/>
{arms}
      <site name="thrust" pos="0 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="thrust" site="thrust" gear="0 0 1 0 0 0" ctrlrange="0 {spec.max_thrust}"/>
{servos}
  </actuator>
</mujoco>
"""


def load_model(spec: QuadrotorSpec = QuadrotorSpec()):
    """The drone's ``mujoco.MjModel``, with the linear drag as damping of the free joint's translation (the compiler sets
    one damping for a joint's six degrees of freedom; the rotation's stays zero)."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(mjcf(spec))
    model.dof_damping[:3] = spec.drag
    model.dof_damping[3:6] = 0.0
    return model

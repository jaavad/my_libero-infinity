"""robot_perturbation.scenic — Layer 3f: Robot joint-reset perturbation.

Perturbs the Panda arm's initial 7-DOF joint configuration in joint space.
This mirrors the semantics used by LIBERO-plus: vary `init_qpos`, not the
robot base pose. The resulting end-effector start pose changes in workspace as
an effect of forward kinematics.

Python usage
────────────
    import scenic
    scenario = scenic.scenarioFromFile(
        "scenic/robot_perturbation.scenic",
        params={"bddl_path": "..."},
    )
    scene, _ = scenario.generate()
    print(scene.params["robot_init_qpos"])  # sampled 7-DOF arm reset
"""

model libero_model

import math

param task = "robot_perturbation"
param bddl_path = ""

param robot_radius_lo = 0.1
param robot_radius_hi = 0.5

_canonical = [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, math.pi / 4.0]
_radius = Range(globalParameters.robot_radius_lo, globalParameters.robot_radius_hi)
_d0 = Range(-1.0, 1.0)
_d1 = Range(-1.0, 1.0)
_d2 = Range(-1.0, 1.0)
_d3 = Range(-1.0, 1.0)
_d4 = Range(-1.0, 1.0)
_d5 = Range(-1.0, 1.0)
_d6 = Range(-1.0, 1.0)
_norm = ((_d0 * _d0 + _d1 * _d1 + _d2 * _d2 + _d3 * _d3 + _d4 * _d4 + _d5 * _d5 + _d6 * _d6) + 1e-12) ** 0.5

param robot_init_radius = _radius
param robot_model = "Panda"
param robot_init_qpos_0 = _canonical[0] + ((_radius * _d0) / _norm)
param robot_init_qpos_1 = _canonical[1] + ((_radius * _d1) / _norm)
param robot_init_qpos_2 = _canonical[2] + ((_radius * _d2) / _norm)
param robot_init_qpos_3 = _canonical[3] + ((_radius * _d3) / _norm)
param robot_init_qpos_4 = _canonical[4] + ((_radius * _d4) / _norm)
param robot_init_qpos_5 = _canonical[5] + ((_radius * _d5) / _norm)
param robot_init_qpos_6 = _canonical[6] + ((_radius * _d6) / _norm)
param robot_init_qpos = [
    globalParameters.robot_init_qpos_0,
    globalParameters.robot_init_qpos_1,
    globalParameters.robot_init_qpos_2,
    globalParameters.robot_init_qpos_3,
    globalParameters.robot_init_qpos_4,
    globalParameters.robot_init_qpos_5,
    globalParameters.robot_init_qpos_6,
]

# Scenic requires an ego object even though robot perturbation only affects the
# simulator-side Panda reset state.
ego = new Object at Vector(0, 0, TABLE_Z), with allowCollisions True

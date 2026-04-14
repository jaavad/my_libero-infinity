"""distractor_perturbation.scenic — Layer 3f: Distractor object injection.

Adds 1-5 non-task distractor objects to the scene to test policy robustness
to visual clutter. Uses the "max slots" pattern: all 5 slots exist in
Scenic's constraint space, but only n_distractors are injected into
MuJoCo via BDDL patching. Inactive slots are harmless — they get sampled
positions but never appear in the simulation.

Task objects remain at canonical positions (no spatial perturbation).

Usage::

    scenario = scenic.scenarioFromFile(
        "scenic/distractor_perturbation.scenic",
        params={"bddl_path": "..."},
    )
    scene, _ = scenario.generate(maxIterations=2000)
    n = int(scene.params["n_distractors"])  # 1-5
    # simulator.setup() auto-detects and injects active distractors
"""

model libero_model

# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
param task = "distractor_perturbation"
param bddl_path = ""
param n_distractors = DiscreteRange(1, 5)
# Distractor-to-task-object clearance (metres, centre-to-centre).
# Footprint diagonals: bowl (0.10×0.10)+distractor (0.08×0.08) ≈ 0.127 m
#                      ramekin (0.08×0.08)+distractor             ≈ 0.113 m
#                      plate   (0.20×0.20)+distractor             ≈ 0.198 m
# Using 0.13 covers bowl/ramekin pairs with a small safety buffer.  Plate is
# slightly under-constrained (0.198 > 0.13), but allowCollisions=True lets
# physics settling resolve minor edge contact.  Values ≥ 0.22 make simultaneous
# placement of 5 distractors infeasible in the 0.70×0.50 m workspace.
param distractor_clearance = 0.13

_dist_cl = globalParameters.distractor_clearance
_n_distractors = globalParameters.n_distractors

# ──────────────────────────────────────────────────────────────────────────────
# Task objects at canonical positions (bowl-on-plate task)
# ──────────────────────────────────────────────────────────────────────────────
bowl = new LIBEROObject with libero_name "akita_black_bowl_1",
                         with asset_class "akita_black_bowl",
                         with width 0.10, with length 0.10, with height 0.06,
                         at Vector(-0.09, 0.0, TABLE_Z)

plate = new LIBEROObject with libero_name "plate_1",
                          with asset_class "plate",
                          with width 0.20, with length 0.20, with height 0.02,
                          at Vector(0.05, -0.02, TABLE_Z)

cream_cheese = new LIBEROObject with libero_name "cream_cheese_1",
                                 with asset_class "cream_cheese",
                                 with width 0.08, with length 0.08, with height 0.05,
                                 at Vector(0.10, 0.12, TABLE_Z)

wine_bottle = new LIBEROObject with libero_name "wine_bottle_1",
                                with asset_class "wine_bottle",
                                with width 0.08, with length 0.08, with height 0.20,
                                at Vector(-0.14, -0.05, TABLE_Z)

# Fixed fixtures on the tabletop for this task. Distractors must avoid them too.
wooden_cabinet = new LIBEROFixture at Vector(0.03, -0.24, TABLE_Z),
                                    with libero_name "wooden_cabinet_1"
flat_stove = new LIBEROFixture at Vector(-0.41, 0.21, TABLE_Z),
                                with libero_name "flat_stove_1"
wine_rack = new LIBEROFixture at Vector(-0.26, -0.26, TABLE_Z),
                               with libero_name "wine_rack_1"

# ──────────────────────────────────────────────────────────────────────────────
# Distractor pool — sourced from the canonical JSON via libero_model.scenic.
# Copy to a local Python list so Scenic analyses it as a static constant rather
# than treating the model-imported name as a Scenic distribution node.
# (DISTRACTOR_POOL is loaded from asset_variants.json["distractor_pool"],
# which mirrors asset_registry.DEFAULT_DISTRACTOR_POOL.)
# ──────────────────────────────────────────────────────────────────────────────
_POOL = list(DISTRACTOR_POOL)

# ──────────────────────────────────────────────────────────────────────────────
# Distractor slots (all 5 always created; only first n_distractors injected)
# ──────────────────────────────────────────────────────────────────────────────
distractors = []
distractor_classes = []
for i in range(5):
    _cls = Uniform(*_POOL)
    distractor_classes.append(_cls)
    d = new LIBEROObject with libero_name f"distractor_{i}",
                             with asset_class _cls,
                             with width 0.08, with length 0.08, with height 0.08,
                             in SAFE_REGION
    distractors.append(d)

# param keyword requires literal identifiers — cannot use f-string keys
param distractor_0_class = distractor_classes[0]
param distractor_1_class = distractor_classes[1]
param distractor_2_class = distractor_classes[2]
param distractor_3_class = distractor_classes[3]
param distractor_4_class = distractor_classes[4]

# ──────────────────────────────────────────────────────────────────────────────
# Clearance constraints. Slot i is enforced only when n_distractors > i.
# ──────────────────────────────────────────────────────────────────────────────
# Distractor-to-task-object
_task_objects = [bowl, plate, cream_cheese, wine_bottle]
for _i, d in enumerate(distractors):
    for t in _task_objects:
        require (_n_distractors <= _i) or ((distance from d to t) > _dist_cl)

# Distractor-to-distractor: compact objects (≤0.08 m footprint) need at least
# their footprint diagonal / 2 ≈ 0.057 m centre-to-centre clearance; 0.06 adds
# a small buffer.  Kept separate from _dist_cl (task-object clearance) because
# distractor–distractor overlaps are less harmful than task-object collisions.
_d2d_clearance = 0.06
for i in range(5):
    for j in range(i + 1, 5):
        require (
            (_n_distractors <= i)
            or (_n_distractors <= j)
            or ((distance from distractors[i] to distractors[j]) > _d2d_clearance)
        )

# Distractor-to-fixture clearance
for _i, d in enumerate(distractors):
    require (_n_distractors <= _i) or ((distance from d to wooden_cabinet) > 0.2680)
    require (_n_distractors <= _i) or ((distance from d to flat_stove) > 0.2622)
    require (_n_distractors <= _i) or ((distance from d to wine_rack) > 0.1642)

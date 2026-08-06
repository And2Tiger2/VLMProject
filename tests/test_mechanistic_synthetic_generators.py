from __future__ import annotations

import hashlib

from vlm_eval.mechanistic_heads.synthetic import (
    SEARCH_COLORS,
    SEARCH_SHAPES,
    SEARCH_TEST_CONJUNCTIONS,
    SEARCH_TRAIN_CONJUNCTIONS,
    fixed_eight_scene,
    render_search_scene,
    render_syndot,
    render_waldo_like_scene,
    syndot_positions,
    waldo_distractor_centers,
    length_matched_nonspatial_answer,
)


def _digest(image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def test_syndot_is_deterministic_and_matches_paper_geometry() -> None:
    first = syndot_positions(7, "example")
    second = syndot_positions(7, "example")
    assert first == second
    image = render_syndot(10, first)
    assert image.size == (336, 336)
    assert _digest(image) == _digest(render_syndot(10, second))


def test_constant_complexity_has_fixed_total_and_exact_masks() -> None:
    four = fixed_eight_scene(seed=2, scene_id="a", red_count=4)
    five = fixed_eight_scene(seed=2, scene_id="a", red_count=5)
    assert len(four.objects) == len(five.objects) == 8
    assert [row["center"] for row in four.objects] == [row["center"] for row in five.objects]
    assert four.masks["changed_pixel"].getbbox() is not None
    assert _digest(four.image) != _digest(five.image)


def test_constant_complexity_relocation_moves_the_category_edit() -> None:
    standard = fixed_eight_scene(
        seed=9, scene_id="relocation", red_count=4, variant="color", relocate=False
    )
    relocated = fixed_eight_scene(
        seed=9, scene_id="relocation", red_count=4, variant="color", relocate=True
    )
    assert standard.objects[4]["center"] != relocated.objects[4]["center"]
    assert [row["center"] for row in standard.objects[:4]] == [
        row["center"] for row in relocated.objects[:4]
    ]


def test_point_search_is_deterministic_with_exact_target_count() -> None:
    first = render_search_scene(
        seed=4,
        scene_id="s",
        target_color="green",
        target_shape="T",
        target_count=2,
    )
    second = render_search_scene(
        seed=4,
        scene_id="s",
        target_color="green",
        target_shape="T",
        target_count=2,
    )
    assert _digest(first.image) == _digest(second.image)
    assert sum(row["class"] == "target" for row in first.objects) == 2
    assert len(first.objects) == 50
    assert first.masks["target"].getbbox() is not None


def test_point_search_uses_exact_paper_colors_shapes_and_conjunctions() -> None:
    assert tuple(SEARCH_COLORS) == (
        "red",
        "green",
        "blue",
        "purple",
        "gray",
        "black",
    )
    assert SEARCH_SHAPES == ("L", "T", "H", "E", "F", "Γ")
    assert SEARCH_TRAIN_CONJUNCTIONS == (
        ("red", "L"),
        ("green", "T"),
        ("blue", "H"),
        ("purple", "E"),
        ("gray", "F"),
        ("black", "Γ"),
    )
    assert SEARCH_TEST_CONJUNCTIONS == (
        ("gray", "L"),
        ("green", "E"),
        ("green", "L"),
        ("red", "E"),
        ("red", "F"),
        ("gray", "T"),
        ("blue", "F"),
        ("black", "H"),
        ("blue", "E"),
        ("red", "H"),
    )
    assert set(SEARCH_TRAIN_CONJUNCTIONS).isdisjoint(SEARCH_TEST_CONJUNCTIONS)
    for shape in SEARCH_SHAPES:
        scene = render_search_scene(
            seed=4,
            scene_id=f"paper-shape-{shape}",
            target_color="red",
            target_shape=shape,
            target_count=1,
        )
        assert scene.masks["target"].getbbox() is not None


def test_waldo_like_generator_is_original_deterministic_and_masked() -> None:
    first = render_waldo_like_scene(
        seed=5, scene_id="w", target_present=True, clutter=8, similarity=3
    )
    second = render_waldo_like_scene(
        seed=5, scene_id="w", target_present=True, clutter=8, similarity=3
    )
    assert _digest(first.image) == _digest(second.image)
    targets = [row for row in first.objects if row["class"] == "target"]
    assert len(targets) == 1
    assert set(targets[0]["features"]) == {
        "striped_torso",
        "round_glasses",
        "pointed_hat",
        "blue_lower",
    }
    assert first.masks["target"].getbbox() is not None
    impostors = [row for row in first.objects if row["class"] == "distractor-incorrect-binding"]
    assert len(impostors) == 1 and not impostors[0]["binding_correct"]


def test_waldo_relocation_keeps_distractors_identical() -> None:
    centers = waldo_distractor_centers(
        seed=9, scene_id="relocate", clutter=8, forbidden_cells=[11, 88]
    )
    assert all(((y // 40) * 10 + x // 40) not in {11, 88} for x, y in centers)
    left = render_waldo_like_scene(
        seed=9,
        scene_id="relocate",
        target_present=True,
        target_cell=11,
        clutter=8,
        distractor_centers=centers,
    )
    right = render_waldo_like_scene(
        seed=9,
        scene_id="relocate",
        target_present=True,
        target_cell=88,
        clutter=8,
        distractor_centers=centers,
    )
    assert [row["center"] for row in left.objects[1:]] == [
        row["center"] for row in right.objects[1:]
    ]
    assert [row["features"] for row in left.objects[1:]] == [
        row["features"] for row in right.objects[1:]
    ]


def test_waldo_verification_changes_only_target_slot() -> None:
    centers = waldo_distractor_centers(
        seed=12, scene_id="verify", clutter=8, forbidden_cells=[44]
    )
    true = render_waldo_like_scene(
        seed=12,
        scene_id="verify",
        target_present=True,
        target_cell=44,
        clutter=8,
        similarity=3,
        distractor_centers=centers,
    )
    impostor = render_waldo_like_scene(
        seed=12,
        scene_id="verify",
        target_present=False,
        target_cell=44,
        clutter=8,
        similarity=3,
        distractor_centers=centers,
    )
    assert true.objects[0]["center"] == impostor.objects[0]["center"]
    assert len(set(true.objects[0]["features"]) - set(impostor.objects[0]["features"])) == 1
    assert true.objects[1:] == impostor.objects[1:]


def test_waldo_decoy_pair_changes_only_one_distractor() -> None:
    centers = waldo_distractor_centers(
        seed=13, scene_id="decoy", clutter=8, forbidden_cells=[22]
    )
    low = render_waldo_like_scene(
        seed=13,
        scene_id="decoy",
        target_present=True,
        target_cell=22,
        clutter=8,
        similarity=1,
        distractor_centers=centers,
    )
    high = render_waldo_like_scene(
        seed=13,
        scene_id="decoy",
        target_present=True,
        target_cell=22,
        clutter=8,
        similarity=1,
        similarity_overrides={1: 3},
        distractor_centers=centers,
    )
    assert low.objects[0] == high.objects[0]
    assert low.objects[1] != high.objects[1]
    assert low.objects[2:] == high.objects[2:]
    assert high.objects[1]["class"] == "distractor-incorrect-binding"


def test_waldo_full_scene_zoom_transforms_image_masks_and_geometry() -> None:
    native = render_waldo_like_scene(
        seed=14,
        scene_id="zoom",
        target_present=True,
        target_cell=44,
        clutter=8,
        scene_zoom=1.0,
    )
    zoomed = render_waldo_like_scene(
        seed=14,
        scene_id="zoom",
        target_present=True,
        target_cell=44,
        clutter=8,
        scene_zoom=1.1,
    )
    assert native.image.size == zoomed.image.size == (400, 400)
    assert native.image.tobytes() != zoomed.image.tobytes()
    assert native.masks["target"].tobytes() != zoomed.masks["target"].tobytes()
    assert native.objects[0]["center"] != zoomed.objects[0]["center"]
    assert zoomed.objects[0]["scene_zoom"] == 1.1
    assert all(0 <= value < 400 for value in zoomed.objects[0]["center"])


def test_length_matched_direct_answer_is_exact_and_nonspatial() -> None:
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            class Encoded:
                input_ids = text.split()
            return Encoded()

    point = "points = one two three four five"
    answer = length_matched_nonspatial_answer(
        Tokenizer(), direct_answer="1", point_answer=point
    )
    assert len(answer.split()) == len(point.split())
    assert "(" not in answer and ")" not in answer

import numpy as np

from scene_memory.mapping.boxer_objects import BoxerObjectMap, fuse_label
from scene_memory.mapping.daaam_mapping import create_label_and_color
from scene_memory.perception.gdino_sam2_detector import class_token_map, nms
from scene_memory.common.labelspace import load_labelspace
from scene_memory.perception.pose_track import ID_STRIDE, ViewPoseTracker, project_centers
from scene_memory.perception.sam3_detector import merge_prompts


def _det(x, cls=0, score=0.9, w=60, h=80):
    return {"xyxy": (x, 200.0, x + w, 200.0 + h), "score": score, "cls": cls, "name": f"c{cls}"}


def _run(tracker, xs, priors_for):
    ids = []
    for i, x in enumerate(xs):
        out = tracker.update("front", [_det(x)], 640, 640, None, priors_for(i, ids))
        ids.append(out[0]["track_id"] if out else None)
    return ids


def test_prior_keeps_id_under_large_ego_motion():
    xs = [100, 105, 110, 400]  # the camera turns: the object jumps 290 px in one detector frame
    plain = _run(ViewPoseTracker(["front"], track_buffer=5), xs, lambda i, ids: {})
    assert plain[2] is not None and plain[3] != plain[2]

    def prior(i, ids):
        return {ids[-1]: (430.0, 240.0)} if i == 3 and ids and ids[-1] else {}

    with_prior = _run(ViewPoseTracker(["front"], track_buffer=5), xs, prior)
    assert with_prior[3] == with_prior[2]
    assert with_prior[3] >= ID_STRIDE


def test_ids_unique_across_views():
    tracker = ViewPoseTracker(["front", "left"])
    a = [tracker.update(v, [_det(100)], 640, 640) for v in ("front", "left") for _ in range(2)]
    assert a[1][0]["track_id"] != a[3][0]["track_id"]


def test_project_centers_in_front_only():
    T = np.eye(4)  # optical frame = world
    objs = [
        {"center": [0.0, 0.0, 2.0], "track_ids": [7]},
        {"center": [0.0, 0.0, -2.0], "track_ids": [8]},
    ]
    uv = project_centers(objs, T, 320, 320, 319.5, 319.5, 640, 640)
    assert set(uv) == {7}
    assert np.allclose(uv[7], (319.5, 319.5))


def test_merge_prompts_groups_classes():
    xyxy = np.array([[0, 0, 10, 10], [0, 0, 10, 11], [50, 50, 60, 60]], float)
    groups = merge_prompts(xyxy, np.array([0.6, 0.8, 0.5]), np.array([0, 1, 0]), 0.7)
    assert groups[0] == (1, {1: 0.8, 0: 0.6})
    assert groups[1] == (2, {0: 0.5})


def test_class_token_map_multiword_and_specials():
    # "[CLS] potted plant . cup . [SEP]"
    offsets = [(0, 0), (0, 6), (7, 12), (12, 13), (14, 17), (17, 18), (0, 0)]
    m = class_token_map(offsets, [(0, 12), (14, 17)])
    assert m[0].tolist() == [False, True, True, False, False, False, False]
    assert m[1].tolist() == [False, False, False, False, True, False, False]


def test_nms_keeps_best_of_overlap():
    xyxy = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], float)
    assert nms(xyxy, np.array([0.4, 0.9, 0.5]), 0.5) == [1, 2]


def test_fuse_label_resists_one_wrong_frame():
    belief = {}
    for _ in range(4):
        belief = fuse_label(belief, {2: 0.9}, 19, 0.7)
    belief = fuse_label(belief, {14: 0.9}, 19, 0.7)
    assert max(belief, key=belief.get) == 2
    assert abs(sum(belief.values()) - 1.0) < 0.05


def test_object_map_fused_name_and_reid():
    names = ["person", "door", "chair", "toilet"]
    m = BoxerObjectMap(class_names=names, label_accuracy=0.7)
    obb = {"center": [1.0, 0.0, 0.0], "size": [0.5, 0.5, 0.8]}
    for _ in range(3):
        m.update([{"obb": obb, "track_id": 5, "cls": 2, "name": "chair", "score": 0.8}], 0.0)
    m.update([{"obb": obb, "track_id": 5, "cls": 3, "name": "toilet", "score": 0.6}], 1.0)
    obj = next(iter(m.objects.values()))
    assert obj["name"] == "chair"
    assert m.fused_label(5) == (2, "chair")
    row = m.to_json()["objects"][0]
    assert list(row["label_belief"])[0] == "chair"
    assert m.track_centers()[0]["track_ids"] == [5]


def test_label_image_with_large_track_ids():
    from pathlib import Path

    ls = load_labelspace(Path(__file__).resolve().parents[1] / "config" / "prompts.txt")
    assert ls.semantic_id(2) > 0
    tracks = [{"xyxy": (10, 10, 30, 30), "track_id": 3 * ID_STRIDE + 4, "cls": 2, "score": 0.9}]
    labels, _ = create_label_and_color(64, 64, tracks, ls)
    assert labels[20, 20] == ls.semantic_id(2)
    assert labels[0, 0] == 0

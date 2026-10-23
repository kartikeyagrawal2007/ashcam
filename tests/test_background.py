import numpy as np

from ashcam.background import BackgroundPlate

H, W = 64, 96


def left_half_mask():
    m = np.zeros((H, W), np.float32)
    m[:, : W // 2] = 1.0
    return m


def test_updates_only_unmasked():
    plate = BackgroundPlate((H, W, 3))
    for _ in range(3):
        plate.update(np.full((H, W, 3), 200, np.uint8), left_half_mask())
    assert (plate.plate[:, -10:] == 200).all()
    assert (plate.plate[:, :10] == 0).all()
    assert plate.known[:, -10:].all()
    assert not plate.known[:, : W // 2].any()


def test_person_halo_excluded():
    plate = BackgroundPlate((H, W, 3))
    plate.update(np.full((H, W, 3), 200, np.uint8), left_half_mask())
    # Columns just right of the mask edge are within 8 px of the person.
    assert not plate.known[:, W // 2 : W // 2 + 8].any()


def test_filled_has_no_holes():
    plate = BackgroundPlate((H, W, 3))
    for _ in range(3):
        plate.update(np.full((H, W, 3), 120, np.uint8), left_half_mask())
    left = plate.filled()[:, : W // 2].astype(np.float32)
    assert abs(left.mean() - 120) <= 10


def test_hole_fill_matches_wall():
    # Person never moved: a person-shaped hole in a plain wall must fill with the wall.
    import cv2
    h, w = 720, 1280
    person = np.zeros((h, w), np.float32)
    cv2.ellipse(person, (640, 430), (170, 300), 0, 0, 360, 1, -1)
    plate = BackgroundPlate((h, w, 3))
    for _ in range(3):
        plate.update(np.full((h, w, 3), 200, np.uint8), person)
    hole = plate.filled()[~plate.known]
    assert hole.min() >= 185 and hole.max() <= 215


def test_nothing_seen_is_not_black():
    plate = BackgroundPlate((H, W, 3))
    plate.update(np.full((H, W, 3), 120, np.uint8), np.ones((H, W), np.float32))
    assert abs(float(plate.filled().mean()) - 120) <= 10


def test_brief_dropout_not_learned():
    plate = BackgroundPlate((H, W, 3))
    person = np.full((H, W, 3), 50, np.uint8)
    person[:, : W // 2] = 250
    for _ in range(3):
        plate.update(person, left_half_mask())
    # Segmenter misses the person for two frames.
    for _ in range(2):
        plate.update(person, np.zeros((H, W), np.float32))
    assert not plate.known[:, :10].any()

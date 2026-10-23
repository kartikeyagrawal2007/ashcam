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


def learnt_plate(wall=200):
    plate = BackgroundPlate((H, W, 3))
    for _ in range(3):
        plate.update(np.full((H, W, 3), wall, np.uint8), left_half_mask())
    return plate


def test_matched_follows_exposure_change():
    # Camera got darker after the room was learnt: remembered room must darken too.
    plate = learnt_plate(200)
    live = np.full((H, W, 3), 150, np.uint8)
    out = plate.matched(live, left_half_mask()).astype(np.float32)
    assert abs(out[:, -20:].mean() - 150) <= 6      # seen area
    assert abs(out[:, :20].mean() - 150) <= 10      # guessed area


def test_matched_follows_colour_shift():
    plate = learnt_plate(200)
    live = np.zeros((H, W, 3), np.uint8)
    live[:] = (150, 200, 230)
    out = plate.matched(live, left_half_mask()).astype(np.float32)
    for ch, want in enumerate((150, 200, 230)):
        assert abs(out[:, :20, ch].mean() - want) <= 12


def test_capture_learns_whole_room():
    plate = BackgroundPlate((H, W, 3))
    room = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    plate.capture(room)
    assert plate.known.all() and np.array_equal(plate.plate, room)


def test_update_relights_hidden_area():
    # You're still sitting there when the camera re-exposes: the remembered
    # wall behind you must follow, even though it can't be seen right now.
    plate = learnt_plate(200)
    plate.filled()          # the app fills (and caches the guess) every frame
    plate.update(np.full((H, W, 3), 150, np.uint8), left_half_mask())
    filled = plate.filled().astype(np.float32)
    assert abs(filled[:, :20].mean() - 150) <= 10

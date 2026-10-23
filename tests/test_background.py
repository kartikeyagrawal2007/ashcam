import numpy as np

from ashcam.background import BackgroundPlate

H, W = 64, 96


def left_half_mask():
    m = np.zeros((H, W), np.float32)
    m[:, : W // 2] = 1.0
    return m


def test_updates_only_unmasked():
    plate = BackgroundPlate((H, W, 3))
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
    plate.update(np.full((H, W, 3), 120, np.uint8), left_half_mask())
    left = plate.filled()[:, : W // 2].astype(np.float32)
    assert abs(left.mean() - 120) <= 10

import numpy as np

from ashcam.fill import LamaFiller


class EchoModel:
    """Stands in for the LaMa session: returns its input image (0..1 RGB) as
    0..255, like the real model, and records what it was given."""
    def __init__(self, paint=None):
        self.paint = paint
        self.calls = []

    def run(self, _outputs, feed):
        img, mask = feed["image"], feed["mask"]
        self.calls.append((img.shape, mask.shape, float(img.max()), float(mask.max())))
        out = img * 255.0
        if self.paint is not None:
            out = np.where(mask > 0.5, np.asarray(self.paint, np.float32).reshape(1, 3, 1, 1), out)
        return [out]


def scene(h=720, w=1280):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    hole = np.zeros((h, w), bool)
    hole[250:600, 520:760] = True
    return img, hole


def test_only_hole_pixels_change():
    img, hole = scene()
    out = LamaFiller(session=EchoModel(paint=(10, 20, 30))).fill(img, hole)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[~hole], img[~hole])


def test_paint_comes_back_in_bgr():
    img, hole = scene()
    out = LamaFiller(session=EchoModel(paint=(10, 20, 30))).fill(img, hole)   # RGB from model
    inner = out[300:550, 560:720].reshape(-1, 3).mean(axis=0)
    assert np.allclose(inner, (30, 20, 10), atol=3)                          # BGR in the frame


def test_model_gets_512_square_unit_range_inputs():
    img, hole = scene()
    model = EchoModel()
    LamaFiller(session=model).fill(img, hole)
    (img_shape, mask_shape, img_max, mask_max), = model.calls
    assert img_shape == (1, 3, 512, 512) and mask_shape == (1, 1, 512, 512)
    assert img_max <= 1.0 and mask_max == 1.0


def test_hole_at_the_frame_edge():
    img, _ = scene()
    hole = np.zeros(img.shape[:2], bool)
    hole[500:720, 0:200] = True
    out = LamaFiller(session=EchoModel(paint=(0, 0, 0))).fill(img, hole)
    assert out[700, 10].max() < 10 and np.array_equal(out[:400], img[:400])


def test_no_hole_is_a_no_op():
    img, _ = scene()
    model = EchoModel()
    out = LamaFiller(session=model).fill(img, np.zeros(img.shape[:2], bool))
    assert np.array_equal(out, img) and model.calls == []


def test_preload_loads_the_model_in_the_background():
    import time
    from ashcam.fill import BackgroundFiller

    class Slow:
        loaded = False

        def _model(self):
            time.sleep(0.05)
            Slow.loaded = True

    bf = BackgroundFiller(Slow())
    bf.preload()
    assert bf.state == "loading"
    time.sleep(0.3)
    assert Slow.loaded and bf.state == "idle"

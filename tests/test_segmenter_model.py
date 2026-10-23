import os
import urllib.request

import pytest

from ashcam import segmenter


def test_interrupted_download_leaves_no_model(tmp_path, monkeypatch):
    target = tmp_path / "models" / "m.tflite"

    def broken(url, path):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise OSError("connection reset")

    monkeypatch.setattr(urllib.request, "urlretrieve", broken)
    with pytest.raises(OSError):
        segmenter.ensure_model(str(target))
    assert not os.path.exists(target)

import numpy as np
from transcriber.pcm import PcmReader
from tests.conftest import to_pcm_bytes, tone


def test_reads_growing_file_from_offset(tmp_path):
    p = tmp_path / "mic.pcm"
    a = tone(1.0)
    p.write_bytes(to_pcm_bytes(a[:8000]))
    r = PcmReader(p, offset_samples=4000)
    assert r.available() == 4000
    got = r.read_all()
    assert len(got) == 4000
    assert np.allclose(got, a[4000:8000], atol=1e-3)
    assert r.offset == 8000
    assert r.read_all().size == 0
    with open(p, "ab") as f:
        f.write(to_pcm_bytes(a[8000:]))
    assert r.available() == 8000
    assert len(r.read_all()) == 8000
    assert r.offset == 16000


def test_missing_file_is_empty(tmp_path):
    r = PcmReader(tmp_path / "nope.pcm")
    assert r.available() == 0
    assert r.read_all().size == 0

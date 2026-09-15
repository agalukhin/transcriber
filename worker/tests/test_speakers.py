import numpy as np
from transcriber.speakers import OnlineClusterer, label, recluster


def v(*xs):
    return np.array(xs, dtype=np.float32)


def test_online_assigns_same_and_new():
    c = OnlineClusterer(similarity=0.5)
    assert c.assign(v(1, 0, 0)) == 0
    assert c.assign(v(0.9, 0.1, 0)) == 0        # close → same speaker
    assert c.assign(v(0, 1, 0)) == 1            # orthogonal → new
    assert c.assign(v(0, 0.95, 0.05)) == 1
    assert c.n_speakers == 2
    assert label(1) == "Speaker 2"


def test_online_restore_from_history():
    c = OnlineClusterer.from_history(similarity=0.5, embeddings=[[1, 0], [0, 1], [1, 0]], labels=[0, 1, 0])
    assert c.n_speakers == 2
    assert c.assign(v(0.99, 0.01)) == 0


def test_recluster_relabels_by_first_appearance():
    embs = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0.9, 0.1]], dtype=np.float32)
    assert recluster(embs, similarity=0.5) == [0, 1, 0, 1, 1]


def test_recluster_trivial_sizes():
    assert recluster(np.zeros((0, 4), dtype=np.float32), 0.5) == []
    assert recluster(np.array([[1, 0]], dtype=np.float32), 0.5) == [0]


def test_recluster_masks_nonfinite_and_zero_rows():
    embs = np.array(
        [[1, 0], [float("nan"), float("nan")], [1, 0], [0, 1], [0, 0]],
        dtype=np.float32,
    )
    assert recluster(embs, similarity=0.5) == [0, 0, 0, 1, 1]


def test_online_assign_handles_nonfinite_embedding():
    c = OnlineClusterer(similarity=0.5)
    assert c.assign(v(1, 0)) == 0
    assert c.assign(v(float("nan"), float("nan"))) == 0
    assert c.n_speakers == 1
    assert c.assign(v(1, 0)) == 0


def test_online_restore_from_history_contiguous_labels():
    c = OnlineClusterer.from_history(similarity=0.5, embeddings=[[1, 0], [0, 1], [1, 0]], labels=[0, 1, 0])
    assert c.n_speakers == 2

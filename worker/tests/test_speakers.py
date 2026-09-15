import numpy as np
from transcriber.speakers import OnlineClusterer, absorb_small_clusters, label, recluster


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


def test_absorb_merges_singleton_into_nearest_cluster():
    # два настоящих спикера по два эмбеддинга + одиночка, близкая к первому
    embs = [[1, 0, 0], [0.95, 0.05, 0], [0, 1, 0], [0, 0.95, 0.05], [0.9, 0.2, 0.1]]
    labels = [0, 0, 1, 1, 2]
    assert absorb_small_clusters(labels, embs, min_size=2) == [0, 0, 1, 1, 0]


def test_absorb_keeps_singleton_when_nothing_is_close():
    embs = [[1, 0, 0], [0.95, 0.05, 0], [0, 1, 0], [0, 0.95, 0.05], [0, 0, 1]]
    labels = [0, 0, 1, 1, 2]
    assert absorb_small_clusters(labels, embs, min_size=2, min_similarity=0.1) == [0, 0, 1, 1, 2]


def test_absorb_relabels_by_first_appearance():
    # одиночка идёт первой: после слияния метки должны пересчитаться по первому появлению
    embs = [[0.9, 0.2, 0.1], [0, 1, 0], [0, 0.95, 0.05], [1, 0, 0], [0.95, 0.05, 0]]
    labels = [2, 1, 1, 0, 0]
    assert absorb_small_clusters(labels, embs, min_size=2) == [0, 1, 1, 0, 0]


def test_absorb_noop_when_every_cluster_is_small():
    embs = [[1, 0], [0, 1]]
    assert absorb_small_clusters([0, 1], embs, min_size=2) == [0, 1]


def test_absorb_survives_zero_and_nonfinite_rows():
    embs = [[1, 0], [0.9, 0.1], [0.0, 0.0], [float("nan"), 1.0]]
    got = absorb_small_clusters([0, 0, 1, 2], embs, min_size=2)
    assert got[:2] == [0, 0] and len(got) == 4

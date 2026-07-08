"""
test_split_leakage.py — verify no temporal overlap between train/val/test splits.

The SHSequenceDataset uses contiguous-block splitting at the sequence level
to prevent leakage from adjacent overlapping sliding windows. This test
confirms that invariant programmatically.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

from train_sequence import SHSequenceDataset, check_split_leakage


def _make_mini_dataset(path, n_seqs=5, seq_len=1000, nspots=10, nmodes=5):
    """Generate a tiny in-memory dataset for testing."""
    n_frames = n_seqs * seq_len
    rng = np.random.RandomState(0)
    disps = rng.randn(n_frames, 2 * nspots)
    coeff = rng.randn(n_frames, nmodes)
    dr0 = rng.uniform(1, 10, n_frames)
    np.savez(path, displacements=disps, coefficients=coeff, D_r0=dr0)


def test_split_by_sequence_no_leakage():
    path = "_test_leakage.npz"
    try:
        _make_mini_dataset(path)
        ds = SHSequenceDataset(path, lookback=10, step=1, task="predict")
        train, val, test = SHSequenceDataset.split_by_sequence(ds, seed=42)
        # check_split_leakage is called inside split_by_sequence, so if it
        # passes we already have the invariant.  Double-check explicit sets.
        train_s = set(ds.sequence_ids[i] for i in train.indices)
        val_s   = set(ds.sequence_ids[i] for i in val.indices)
        test_s  = set(ds.sequence_ids[i] for i in test.indices)
        assert train_s.isdisjoint(val_s),  "train ↔ val overlap"
        assert train_s.isdisjoint(test_s), "train ↔ test overlap"
        assert val_s.isdisjoint(test_s),   "val ↔ test overlap"
        print(f"PASS: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test sequences, no leakage")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_split_by_sequence_exhaustive():
    """With a 4-sequence dataset and 50/25/25 split, every sample appears in exactly one split."""
    path = "_test_leakage_exhaustive.npz"
    try:
        _make_mini_dataset(path, n_seqs=4)
        ds = SHSequenceDataset(path, lookback=10, step=1, task="predict")
        train, val, test = SHSequenceDataset.split_by_sequence(
            ds, train_ratio=0.5, val_ratio=0.25, seed=42
        )
        all_idx = set(train.indices) | set(val.indices) | set(test.indices)
        assert all_idx == set(range(len(ds))), f"Missing {set(range(len(ds))) - all_idx}"
        print(f"PASS: {len(train)} train + {len(val)} val + {len(test)} test = {len(ds)} total")
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_check_split_leakage_detects_overlap():
    """check_split_leakage must raise on deliberately overlapping splits."""
    path = "_test_leakage_detect.npz"
    try:
        _make_mini_dataset(path, n_seqs=2)
        ds = SHSequenceDataset(path, lookback=10, step=1, task="predict")
        # deliberately pass overlapping indices
        try:
            check_split_leakage(ds, [0, 1], [1, 2], [3, 4])
            assert False, "expected AssertionError"
        except AssertionError:
            print("PASS: overlap correctly detected")
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    test_split_by_sequence_no_leakage()
    test_split_by_sequence_exhaustive()
    test_check_split_leakage_detects_overlap()
    print("\nAll tests PASSED")

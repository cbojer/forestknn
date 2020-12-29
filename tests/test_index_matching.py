from forestknn import matching_leaf_indexes
import numpy as np


def test_it_finds_matching_leaf_indexes():
    data = np.array([1, 2, 3, 4, 5])
    assert (matching_leaf_indexes(data, 2) == np.array([0, 1, 0, 0, 0])).all()


def test_it_finds_multiple_matching_indexes():
    data = np.array([1, 2, 3, 1, 5])
    assert (matching_leaf_indexes(data, 1) == np.array([1, 0, 0, 1, 0])).all()


def test_it_finds_no_matching_indexes():
    # Shouldn't really happen..
    data = np.array([1, 2, 3, 4, 5])
    assert (matching_leaf_indexes(data, 6) == np.array([0, 0, 0, 0, 0])).all()
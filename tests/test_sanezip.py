import pytest
from projects.shared.utils import sanezip
# thanks claude!
def test_sanezip_equal_length_lists():
    result = list(sanezip([1, 2, 3], ['a', 'b', 'c']))
    assert result == [(1, 'a'), (2, 'b'), (3, 'c')]

def test_sanezip_empty_iterables():
    result = list(sanezip([], []))
    assert result == []

def test_sanezip_different_iterable_types():
    result = list(sanezip(range(3), "abc"))
    assert result == [(0, 'a'), (1, 'b'), (2, 'c')]

def test_sanezip_first_longer():
    with pytest.raises(ValueError, match="Iterables have different lengths"):
        list(sanezip([1, 2, 3], [1, 2]))
    with pytest.raises(ValueError, match="Iterables have different lengths"):
        list(sanezip({1, 2, 3}, {1, 2}))

def test_sanezip_second_longer():
    with pytest.raises(ValueError, match="Iterables have different lengths"):
        list(sanezip([1, 2], [1, 2, 3]))
    with pytest.raises(ValueError, match="Iterables have different lengths"):
        list(sanezip({1, 2}, {1, 2, 3}))

def test_sanezip_with_generators():
    def gen1():
        yield from range(3)
    
    def gen2():
        yield from 'abc'
    
    result = list(sanezip(gen1(), gen2()))
    assert result == [(0, 'a'), (1, 'b'), (2, 'c')]

def test_sanezip_single_elements():
    result = list(sanezip([1], ['a']))
    assert result == [(1, 'a')]

def test_sanezip_with_tuples():
    result = list(sanezip((1, 2), (3, 4)))
    assert result == [(1, 3), (2, 4)]

def test_sanezip_with_sets():
    # Note: Sets are unordered, so we need to ensure the lengths are equal
    # but can't test exact element pairing
    result = list(sanezip({1}, {2}))
    assert len(result) == 1
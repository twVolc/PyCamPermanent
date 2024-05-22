import pytest
from pycam.utils import truncate_path

normal_test_data = [
    ("tiny path", 10, "tiny path"),
    ("this is a short path", 10, "...short path"),
    ("this is a short path", 25, "this is a short path"),
    ("this is a much, much longer path", 25, "... a much, much longer path"),
    ("this is a much, much longer path", 40, "this is a much, much longer path"),
]

@pytest.mark.parametrize("path, max_length, expected", normal_test_data)
def test_truncate_path_normal(path, max_length, expected):
    result = truncate_path(path, max_length)
    assert result == expected

error_test_data = [
    ("", 20, "path length should be greater than 0"),
    ("normal string", 0, "max_length should be greater than 0"),
    ("another normal string", -1, "max_length should be greater than 0"),
]

@pytest.mark.parametrize("path, max_length, msg", error_test_data)
def test_truncate_path_error(path, max_length, msg):
    with pytest.raises(ValueError, match=msg):
        truncate_path(path, max_length)
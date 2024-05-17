import pytest
from src import va_am
import sys
import numpy as np

@pytest.mark.parametrize(
    "size, ratio_w_h, expected_size",
    [
        (100, 0.5, (5, 20)),
        (100, -1.5, (1, 100)),
        (100, 1, (10, 10)),
        (100, 3.8, (2, 50)),
        (True, 1, (1, 1)),
        (sys.maxsize, 0, (1, sys.maxsize)),
        (100, float('inf'), (1, 100)),
        (100, sys.float_info.max, (1, 100)),
        (100, np.nan, (1, 100))
    ],
)
def test_square_dims_normal(size, ratio_w_h, expected_size):
    assert va_am.square_dims(size, ratio_w_h) == expected_size
    return

@pytest.mark.parametrize(
    "size, ratio_w_h",
    [
        (100.0, 0.5),
        (0, 0.5),
        (sys.float_info.max, 0.5),
        (float('inf'), 0.5),
    ],
)
def test_square_dims_valueerror(size, ratio_w_h):
    with pytest.raises(ValueError):
        va_am.square_dims(size, ratio_w_h)
    return

@pytest.mark.parametrize(
    "size, ratio_w_h",
    [
        ('a', 0.5),
        ((), 0.5),
        (test_square_dims_normal, 0.5),
        ([], 0.5),
        (100, []),
        (100, 'a')
    ],
)
def test_square_dims_typeerror(size, ratio_w_h):
    with pytest.raises(TypeError):
        va_am.square_dims(size, ratio_w_h)
    return
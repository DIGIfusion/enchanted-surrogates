import numpy as np
import pytest
import itertools 
import sys, os 
import pandas as pd

sys.path.append(os.getcwd() + "/src")

from common import data_split
# TODO: figure out how this works :D
valid_sizes = [0.1, 0.2, 0.3, 0.4]
test_sizes = [0.1, 0.2, 0.3, 0.4]
valid_test = list(itertools.product(valid_sizes, test_sizes))
@pytest.mark.parametrize("valid_size,test_size", valid_test)
def test_data_split(valid_size: float, test_size: float):
    # valid_size= 0.1
    # test_size = 0.1
    df = pd.DataFrame(data=np.arange(100).reshape(50,2), columns=['a','b'])
    train, valid, test, pool = data_split(df, valid_size, test_size)

    assert len(train) == 0 
    # check that len(valid) and len(test) are same proprotion as their argument
    # and that len(pool) is the remaining proporiton 

    assert (len(valid) <= int(len(df) * valid_size) + 1 )  & (len(valid) >= int(len(df) * valid_size) - 1)
    assert (len(test) <= int(len(df) * test_size) + 1) & (len(test) >= int(len(df) * test_size) - 1)
    assert len(pool) == len(df) - len(valid) - len(test)
    # check that the sum of the lengths of the splits is equal to the length of the original dataframe
    assert len(train) + len(valid) + len(test) + len(pool) == len(df)
    

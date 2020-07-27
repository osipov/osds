from osds.utils import ObjectStorageDataset
from torch.utils.data import DataLoader
import pytest
import pandas as pd
import numpy as np

path_cal_housing = "file://data/california_housing_test.csv"
df_pd = pd.read_csv('data/california_housing_test.csv')



class TestObjectShape(object):

  ### test dimensions
    def test_dimensions(self):
        object_name = ObjectStorageDataset(path_cal_housing)
        expected = len(df_pd)
        actual = object_name.dataset_size
        message = "object length {0} and actual object length {1} doesn't match".format(expected, actual)
        assert actual == expected, message

class TestBatchSize(object):

  ### test when batch_size is less than or equal to zero
    def test_when_input_less_than_zero(self):
        batch_input = 10
        object_name = ObjectStorageDataset(path_cal_housing, batch_size= batch_input)
        actual = object_name.batch_size
        message = "The batch size must be specified as a positive (greater than 0) integer"  
        assert actual > 0, message


  ### test when batch size is not integer
    def test_when_input_not_int(self):
          batch_input = 15
          object_name = ObjectStorageDataset(path_cal_housing, batch_size= batch_input)
          assert type(object_name.batch_size) is int, "The batch size must be an integer"

  
  ### test when batch size input is None
    def test_when_input_is_None(self):
        batch_input = None
        expected = 3000
        object_name = ObjectStorageDataset(path_cal_housing, batch_size= batch_input)
        actual = object_name.batch_size
        message = "object_name.batch_size should return the int {0}, but it actually returned {1}".format(expected, actual)    
        assert actual == expected, message

  ### test when batch size input is more than the size of dataset
    def test_when_input_is_more_than_size_of_dataset(self):
        batch_input = 10
        max = 3000
        object_name = ObjectStorageDataset(path_cal_housing, batch_size= batch_input)
        actual = object_name.batch_size
        message = "object_name.batch_size can not exceed more than the size of dataset"
        assert actual <= max, message

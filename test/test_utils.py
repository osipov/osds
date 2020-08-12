from osds.utils import ObjectStorageDataset
from torch.utils.data import DataLoader
import pytest
import pandas as pd
import numpy as np

object_name1 = ObjectStorageDataset(f"gcs://gs://storage_bucket01/BicycleWeather.csv", batch_size = 20)


batch = iter(DataLoader(object_name1))
batch1 = next(batch)


class TestObjectShape(object):


    def test_dimensions(self):     
        expected = 1340
        actual = object_name1.dataset_size
        message = "object length {0} and actual object length {1} doesn't match".format(expected, actual)
        assert actual == expected, message

class TestBatchSize(object):


    def test_when_input_less_than_zero(self):      
        actual = object_name1.batch_size
        expected = 20
        max = 1340 
        message = "The batch size must be specified as a positive (greater than 0) integer"  
        message1 = "object_name.batch_size should return the int {0}, but it actually returned {1}".format(expected, actual)
        message2 = "object_name.batch_size can not exceed more than the size of dataset"
        assert actual > 0, message
        assert type(object_name1.batch_size) is int, "The batch size must be an integer"
        assert actual == expected, message1
        assert actual <= max, message2 

               
class TestObjectDataType(object):

  ### test default data type - it should be float64
    def test_default_dtype(self):
        expected = 'torch.int64'
        actual = str(batch1.dtype)
        message = "expected object dtype {0} and actual object dtype {1} doesn't match".format(expected, actual)
        assert actual == expected, message
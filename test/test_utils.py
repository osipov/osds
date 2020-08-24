from osds.utils import ObjectStorageDataset
from torch.utils.data import DataLoader
import pytest
import pandas as pd
import numpy as np

object_name1 = ObjectStorageDataset(f"gcs://gs://storage_bucket01/BicycleWeather.csv", storage_options = {'anon' : False }, batch_size= 20, iterations = 20,  eager_load_batches= False, dtype = 'int64')


class TestBatchSize(object):


    def test_input_batchsize(self):      
        actual = object_name1.batch_size
        expected = 2000
        max = object_name1.dataset_size
        message = "The batch size must be specified as a positive (greater than 0) integer"  
        message1 = "object_name.batch_size should return the int {0}, but it actually returned {1}".format(expected, actual)
        message2 = "object_name.batch_size can not exceed more than the size of dataset"
        assert actual > 0, message
        assert type(object_name.batch_size) is int, "The batch size must be an integer"
        assert actual == expected, message1
        assert actual <= max, message2 
               

class TestIterations(object):       

   def test_iterations(self):
      actual = 20
      expected = object_name1.iterations
      message = "object_name.iterations should match with entered number"
      message1 = "object_name.iterations should be integer"
      assert actual == expected, message
      assert type(actual) is int, message1

               
class TestObjectDataType(object):

  ### test default data type - it should be float64
    def test_default_dtype(self):
        expected = 'int64'
        actual = str(object_name1.dtype)
        message = "expected object dtype {0} and actual object dtype {1} doesn't match".format(expected, actual)
        assert actual == expected, message
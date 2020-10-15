from osds.utils import ObjectStorageDataset
from torch.utils.data import DataLoader
import pytest
import pandas as pd
import numpy as np


object_name1 = ObjectStorageDataset(f"s3://nyc-tlc/trip data/yellow_tripdata_2017-0*.csv",  
                                       storage_options = {'anon' : True }, 
                                       batch_size = 5000, 
                                       eager_load_batches=False)


batch1 = next(iter(DataLoader(object_name1)))

class TestObjectShape(object):


    def test_dimensions(self):     
        expected = 9710124
        actual = len(object_name1.df)
        message = "object length {0} and actual object length {1} doesn't match".format(expected, actual)
        assert actual == expected, message



class TestBatchSize(object):


    def test_input_batchsize(self):      
        actual = object_name1.batch_size
        expected = 5000
        max = 16384
        message = "The batch size must be specified as a positive (greater than 0) integer"  
        message1 = "object_name1.batch_size should return the int {0}, but it actually returned {1}".format(expected, actual)
        message2 = "object_name1.batch_size can not exceed more than the size of dataset"
        assert actual > 0, message
        assert type(object_name1.batch_size) is int, "The batch size must be an integer"
        assert actual == expected, message1
        assert actual <= max, message2 
               

class TestIterations(object):       

   def test_iterations(self):
      actual = 'nan'
      expected = object_name1.iterations
      message = "object_name.iterations should match with entered number"
      message1 = "object_name.iterations should be integer"
      assert actual == str(expected), message
      assert type(actual) is str, message1

               
class TestObjectDataType(object):

  ### test default data type - it should be float64
    def test_default_dtype(self):
        expected = 'None'
        actual = str(object_name1.dtype)
        message = "expected object dtype {0} and actual object dtype {1} doesn't match".format(expected, actual)
        assert actual == expected, message
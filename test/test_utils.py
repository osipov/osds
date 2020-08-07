from osds.utils import ObjectStorageDataset
from torch.utils.data import DataLoader

object_name1 = ObjectStorageDataset(f"gcs://gs://cloud-training-demos/taxifare/large/taxi-train*.csv", batch_size= 2000, iterations = 20, storage_options = {'anon' : False }, eager_load_batches=True)

batch1 = next(iter(DataLoader(object_name1)))


class TestObjectShape(object):


    def test_dimensions(self):     
        expected = 21568243
        actual = object_name.dataset_size
        message = "object length {0} and actual object length {1} doesn't match".format(expected, actual)
        assert actual == expected, message

class TestBatchSize(object):


    def test_when_input_less_than_zero(self):      
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


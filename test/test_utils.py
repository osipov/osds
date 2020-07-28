### the dictionary of expected outputs
dict1 = {
    'object_name.dtype' : 'None',
    'object_name1.dtype' : 'float16',
    'object_name.dataset_size' : 21568243,
    'object_name1.batch_size' : 100, 
    'object_name.batch_size' : 21568243,
    'object_name1.iterations' : 20, 
    'object_name.iterations' : 'nan'
}

class TestObjectShape(object):


    def test_dimensions(self):     
        expected = 21568243
        actual = dict1['object_name.dataset_size']
        message = "object length {0} and actual object length {1} doesn't match".format(expected, actual)
        assert actual == expected, message

class TestBatchSize(object):


    def test_when_input_less_than_zero(self):      
        actual = dict1['object_name1.batch_size']
        expected = 100
        max = dict1['object_name.dataset_size']
        message = "The batch size must be specified as a positive (greater than 0) integer"  
        message1 = "object_name.batch_size should return the int {0}, but it actually returned {1}".format(expected, actual)
        message2 = "object_name.batch_size can not exceed more than the size of dataset"
        assert actual > 0, message
        assert type(dict1['object_name.batch_size']) is int, "The batch size must be an integer"
        assert actual == expected, message1
        assert actual <= max, message2 
               

class TestIterations(object):       
   
   def test_default_iterations(self):
        actual = 'nan'
        expected = str(dict1['object_name.iterations'])
        message = "default object_name.iterations should be nan"
        assert actual == expected, message

   def test_iterations(self):
      actual = 20
      expected = dict1['object_name1.iterations']
      message = "object_name.iterations should match with entered number"
      message1 = "object_name.iterations should be integer"
      assert actual == expected, message
      assert type(actual) is int, message1

class TestObjectDataType(object):

    def test_default_dtype(self):
        expected = 'None'
        actual = str(dict1['object_name.dtype'])
        message = "expected object dtype {0} and actual object dtype {1} doesn't match".format(expected, actual)
        assert actual == expected, message

    def test_userinput_dtpye(self):
      expected = 'float16'
      actual = str(dict1['object_name1.dtype'])
      message = "expected object dtype {0} and actual object dtype {1} doesn't match".format(expected, actual)
      assert actual == expected, message
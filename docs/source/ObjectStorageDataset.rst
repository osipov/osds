Object_Storage_Dataset Class
===================================

.. raw:: html

    <style> .red {color:#D0312D; font-weight:bold; font-size:16px} </style>

.. role:: red

:red:`PyTorch` Iterable Dataset with support for a variety of object (and file) stores.Wraps :red:`fsspec` to enable support for *`s3`, `gcs`, `blob`, `hdfs`*, and other storage options.
Can layer (buffered) text-mode and compression over any file-system, which are typically binary-only.
These instances are safe to serialize, as the low-level file object is not created until invoked using `with`.


.. testcode::

    def __init__(self, glob, storage_options=None,
                 batch_size=None,dtype = None,
                 iterations=None, eager_load_batches=None,
                 its_in_node_memory=True, fits_in_cluster_memory=True,  
                 replicas=1, worker=0, cache_dir=None, tensor_cache_size=1,
	         partition_cache_size=None, batch_cache_size=1)


Glob
----------------
*The protocol handler string specifying a single or multiple files. (str, required)*

To instantiate the :red:`ObjectStorageDataset`, you must specify a URL-style path (similar to a Unix glob
string) pointing to the location of your CSV-formatted dataset.

Examples:

.. testcode::

	object_name = ObjectDataStorage(glob = f"s3://aws-bucket-name/
					directory/files*.extension")
	
	object_name2 = ObjectDataStorage(f"gcs://gcs-bucket-name/
					 folder/file.extension")

Let's see another example from aws, if you have configured the :red:`BUCKET_ID` and :red:`AWS_DEFAULT_REGION` environment variables for the S3 bucket containing your
cleaned up dataset, you can instantiate the class using:


.. testcode::
        
        BUCKET_ID = os.environ['BUCKET_ID']
	AWS_DEFAULT_REGION = os.environ['AWS_DEFAULT_REGION']

	BATCH_SIZE = 1_048_576 // = 2 ** 20

	train_ds = ObjectStorageDataset(f"s3://dc-taxi-{BUCKET_ID}-
	{AWS_DEFAULT_REGION}/csv_vacuum/part*.csv", batch_size=BATCH_SIZE)


    

Batch Size
-----------------
Number of examples that should be returned per batch for every call to `__iter__` method. (None or int, optional)

The :red:`batch_size` parameter used in the example is required when using :red:`ObjectStorageDataset` with out-of-memory datasets. 
By default, the :red:`ObjectStorageDataset` is designed to instantiate in the shortest amount of time possible in order
to start the iterations of gradient descent.

The :red:`batch_size` should be integer, positive number. 

-----------------------------------------------------------------------------------------------------------------------------------------


|
|
|
|

DataType (dtype)
------------------




Eager_load_batches
-------------------





Object_Storage_Dataset Class Other Parameters
==============================================




fits_in_node_memory
-----------------------



replicas
---------------




worker
--------------




cache_dir
---------------



tensor_cache_size
--------------------




partition_cache_size
------------------------





batch_cache_size
----------------------






Object Methods
======================







 

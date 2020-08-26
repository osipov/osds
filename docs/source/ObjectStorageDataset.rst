Object_Storage_Dataset Class
===================================

.. contents::

.. raw:: html

    <style> .red {color:#D0312D; font-size:14px} </style>

.. role:: red

:red:`PyTorch` Iterable Dataset with support for a variety of object (and file) stores.Wraps :red:`fsspec` to enable support for *`s3`, `gcs`, `blob`, `hdfs`*, and other storage options.
Can layer (buffered) text-mode and compression over any file-system, which are typically binary-only.
These instances are safe to serialize, as the low-level file object is not created until invoked using `with`.


.. testcode::

    def __init__(self, glob, storage_options=None, batch_size=None,dtype = None,
                 iterations=None, eager_load_batches=None, its_in_node_memory=True, 
		 fits_in_cluster_memory=True,replicas=1, worker=0, cache_dir=None, 
		 tensor_cache_size=1, partition_cache_size=None, batch_cache_size=1)





Glob
----------------
*The protocol handler string specifying a single or multiple files. (str, required)*

To instantiate the :red:`ObjectStorageDataset`, you must specify a URL-style path (similar to a Unix glob
string) pointing to the location of your CSV-formatted dataset.

Examples:

.. testcode::

	object_name = ObjectDataStorage(glob = f"s3://aws-bucket-name/directory/files*.extension")
	
	object_name2 = ObjectDataStorage(f"gcs://gcs-bucket-name/folder/file.extension")

Let's see another example from aws, if you have configured the :red:`BUCKET_ID` and :red:`AWS_DEFAULT_REGION` environment variables for the S3 bucket containing your
cleaned up dataset, you can instantiate the class using:


.. testcode::
        
        BUCKET_ID = os.environ['BUCKET_ID']
	AWS_DEFAULT_REGION = os.environ['AWS_DEFAULT_REGION']

	BATCH_SIZE = 1_048_576 // = 2 ** 20

	train_ds = ObjectStorageDataset(f"s3://dc-taxi-{BUCKET_ID}-
	{AWS_DEFAULT_REGION}/csv_vacuum/part*.csv", batch_size=BATCH_SIZE)



Storage_Options:
--------------------

If not specified or None, then assumed to be {'anon': True} which uses unauthenticated access to object storage. (None or dict, optional)

To specify storage platform specific authentication, use {'anon': False} instead. Also, use this dict to specify storage platform specific options, e.g. {'anon': False, 'project': 'myproject'}

.. testcode::

	bicycle_weather = ObjectStorageDataset(f"gcs://gs://bucket_name/BicycleWeather.csv", storage_options = {'anon' : False }, batch_size = 20)

  

Batch Size
-----------------
Number of examples that should be returned per batch for every call to `__iter__` method. (None or int, optional)

The :red:`batch_size` parameter used in the example is required when using :red:`ObjectStorageDataset` with out-of-memory datasets. 
By default, the :red:`ObjectStorageDataset` is designed to instantiate in the shortest amount of time possible in order
to start the iterations of gradient descent.

The :red:`batch_size` should be integer, positive number. 



DataType (dtype)
------------------

Specification of the data type to use when loading the dataset in memory. (None, str, or dict, optional)

When `None`, the widest possible data type is used for numeric data which prevents loss of information but uses extra memory. When using an `str` the same `dtype` is used for all numeric columns in the dataset. When using a Python `dict`, use keys that match the column names (or column indicies) from the source dataset, and values that map to Python native data types or NumPy compatible dtype names. For example: *{'col_a': 'Int64', 'col_b': 'int32', 'col_c': 'np.float16'}*. 

you can specify the PyTorch data type for :red:`ObjectStorageDataset` to use on column by column basis
as shown here:


.. testcode::

	train_ds = ObjectStorageDataset(f"s3://dc-taxi-{BUCKET_ID}-{AWS_DEFAULT_REGION}/csv_vacuum/part*.csv", batch_size=BATCH_SIZE,
					dtype={'fareamount': 'float16',
					'origin_block_latitude': 'float16',
					'origin_block_longitude':'float16',
					'destination_block_latitude':'float16',
					'destination_block_longitude':'float16'})


In the cases where all the columns in the dataset use a common dtype, you can use a Python str
instead of the dict from the previous example, simplifying this to:


.. testcode::


	train_ds = ObjectStorageDataset(f"s3://dc-taxi-{BUCKET_ID}-{AWS_DEFAULT_REGION}/csv_vacuum/part*.csv", batch_size=BATCH_SIZE, dtype='float16')


Eager_load_batches
-------------------

int on whether to pre-load partitions from object storage to memory. (None or `Boolean`, optional)

When not specified, datasets that originate from a local filesystem (glob protocol starts with file://) and fit in memory or cluster memory are pre-loaded to cache. Datasets that do not originate from a local filesystem (for example glob protocol starts with s3:// or gcs://), are not pre-loaded by default unless eager_load_batches is set to True. Avoid setting eager_load_batches to True for datasets that do not fit in the node and cluster memory since this may lead to out of memory conditions.

you can use the eager_load_batches named parameter when instantiating the
:red:`ObjectStorageDataset`. For example when using:


.. testcode::


	train_ds = ObjectStorageDataset(f"s3://dc-taxi-${BUCKET_ID}-${AWS_DEFAULT_REGION}/test/part*.csv", eager_load_batches=True)




fits_in_node_memory
-----------------------

Specifies whether the dataset fits in the memory of the node executing this process. (`Boolean`, optional)

When not specified, assumed to be `True`, and if the dataset originates from a local file system, the dataset is pre-loaded to in memory cache. For additional details see `eager_load_batches`.


worker
--------------

When `fits_in_cluster_memory` is `True` and `fits_in_node_memory` is `False`, specifies the number of the worker in the cluster executing this process.(`int`, optional)


The value must be an integer in the range from 0 (inclusive) to `replicas` (exclusive). The selection of the worker number specifies the partitions of the dataset assigned to this process. For more see `fits_in_cluster_memory`.


.. testcode::


	bicycle_weather = ObjectStorageDataset(f"gcs://gs://bucket/BicycleWeather.csv", storage_options = {'anon' : False }, batch_size = 20, worker = 4)

cache_dir
---------------

Location on the runtime's local file system used to store a local cache of the objects (or files) downloaded from the location specified by `glob`. ( None or str, optional)

When not specified, set to `1`, to support the default use case of a dataset that fits in the memory of a node. To avoid out of memory problems, do not specify this cache size unless the batch size is large compared to the size of dataset partitions processed by this node.

tensor_cache_size
--------------------

Specifies the number of PyTorch tensor instances cached in memory.(`int`, optional)

When not specified, set to `1`, to support the default use case of a dataset that fits in the memory of a node. To avoid out of memory problems, do not specify this cache size unless the batch size is large compared to the size of dataset partitions processed by this node.

partition_cache_size
------------------------

Specifies the number of dataset partitions (objects from object storage) cached in memory. (`int`, optional)When not specified or `None`, assumed to be unlimited to support the use case where the entire dataset fits in the memory of the this node.


batch_cache_size
----------------------

Specifies the number of batches cached in memory. (`int`, optional) When not specified, set to `1` to support the default use case when the entire dataset fits in the memory of the node and the use case where at most 1 batch fits in the memory of this node.


replicas
---------------

When `fits_in_cluster_memory` is `True` and `fits_in_node_memory` is `False`, specifies the total number of nodes in the cluster executing this process. (`int`, optional)








 

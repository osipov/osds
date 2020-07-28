Introduction:
---------------------

.. raw:: html

    <style> .red {color:#D0312D; font-weight:bold; font-size:16px} </style>

.. role:: red

The :red:`ObjectStorageDataset` provides support for tensor-based, out-of-memory datasets for the
iterable-style interface. The :red:`ObjectStorageDataset` is not available by default when you install
PyTorch, so you need to install it separately in your Python environment using:

.. testcode::

    pip install osds

and once installed, import the class in your runtime using:


.. testcode::

    from osds.utils import ObjectStorageDataset



The :red:`ObjectStorageDataset` class provides a standard PyTorch interface to datasets stored in the CSV
format, regardless of whether they are located in public cloud object storage or on your local file
system. For every call to the :red:`__iter__` method of the class, the result is a PyTorch tensor of the
numeric values from the CSV based dataset.

The tensor returned by :red:`ObjectStorageDataset` must be separated into the label and the features needed to perform an
iteration of gradient descent.For example:


.. testcode::

     object_name = ObjectStorageDataset('./path/')

     batch = next(iter(DataLoader(object_name)))
     labels, features = batch[:, 0], batch[:, 1:]


Different path-url initiators are used for different storage options. For example, the path url of the file residing on the local storage system intiate from :red:`"file://"`. Please review the following table for more details. 

============================== ==================================================
	path-url            			Description
============================== ==================================================
	:red:`file://`     	Files residing on the local file system
	:red:`f"s3://`     	Amazon-S3   
	:red:`abfs://`     	Azure Blob storage
	:red:`f"gcs://`    	Google Cloud Storage
============================== ==================================================

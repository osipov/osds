**Iterable object from: Amazon-S3**
====================================

.. raw:: html

    <style> .red {color:#D0312D; font-weight:bold; font-size:16px} </style>

.. role:: indexred
.. role:: red

.. toctree::
   :maxdepth: 2




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


**Object Storage Dataset**
=========================================

.. contents::


To get an an iterable style object from large/out-of-memory CSV files stored in local file-system or Cloud storage platforms like Amazon-S3, Google cloud storage, and Azure blob storage.

.. raw:: html

    <style> .red {color:#D0312D; font-weight:bold; font-size:12px} </style>

.. role:: red

The :red:`ObjectStorageDataset` provides support for tensor-based, out-of-memory datasets for the
iterable-style interface. The :red:`ObjectStorageDataset` is not available by default when you install
PyTorch, so you need to install it separately in your Python environment using:

.. testcode::

    pip install osds

and once installed, import the class in your runtime using:


.. testcode::

    from osds.utils import ObjectStorageDataset

Introduction
---------------

.. toctree::
   :maxdepth: 2


   Introduction


Python API
----------------
.. toctree::
   :maxdepth: 2


   ObjectStorageDataset
   



Use Cases
----------------
.. toctree::
   :maxdepth: 2

   linux
   gcs

Indices
----------------
.. toctree::
   :maxdepth: 2
   
   license

* :ref:`search`


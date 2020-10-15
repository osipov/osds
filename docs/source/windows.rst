Iterable object from: local file system (Windows)
=========================================================

.. raw:: html

    <style> .red {color:#D0312D; font-size:14px} </style>

.. role:: indexred
.. role:: red

.. toctree::
   :maxdepth: 2


Our file is already residing on the windows file system, so we will start our url from the :red:`file://` intiator. Let's instantiate the :red:`ObjectStorageDatast`. The type of ``cal_housing`` object is :red:`osds.utils.ObjectStorageDataset`.

.. testcode::

    from osds.utils import ObjectStorageDataset
    from torch.utils.data import DataLoader

    cal_housing =  ObjectStorageDataset("file://C:\\Users\\laayt\\OneDrive\\Pictures\\california_housing_test.csv", batch_size = 10)

    type(cal_housing)
    >> osds.utils.ObjectStorageDataset

    

Batch object type is :red:`torch.Tensor`. The batch size is 10, and number of columns are 9, so the shape of would be **[10, 9]**.

.. testcode::


    batch = next(iter(DataLoader(cal_housing)))

    type(batch)
    >> torch.Tensor

    batch.shape
    >> torch.Size([1, 10, 9])


We also split database based on labels and features. Labels has only one row and 9 columns, so the shape of labels would be **[1, 9]**. The shape of features of each batch would be **[9, 9]**.


.. testcode::

    labels, features = batch[:, 0], batch[:, 1:]

    labels.shape
   >> torch.Size([1, 9])

   features.shape
   >> torch.Size([1, 9, 9])


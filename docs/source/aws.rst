Iterable object from: Amazon Cloud Storage
===========================================

.. raw:: html

    <style> .red {color:#D0312D;  font-size:14px} </style>
    <style> .red {color:#D0312D;  font-size:14px} </style>

.. role:: indexred
.. role:: red

.. toctree::
   :maxdepth: 2

Try to import data from S3 on google colab:

https://colab.research.google.com/drive/1NIzKmm0Dr0o2InZvgpyvu5xWvOtujnfo?usp=sharing


All of the capabilities of :red:`ObjectDataStorage` can also be applied to datasets that reside in serverless object storage services like S3, Google Cloud storage, and Azure. To access datasets from Amazon cloud storage, the faster and the efficient way is to directly call. *f"s3://"*  can be used to call datasets directly from the Google Cloud Storage. 
 
.. testcode::

    from osds.utils import ObjectStorageDataset
    from torch.utils.data import DataLoader


    osds_object = ObjectStorageDataset(f"s3://laaypublicstorage01/BicycleWeather.csv",  storage_options = {'anon' : False }, batch_size = 20, eager_load_batches=False)

    type(osds_object)
    >> osds.utils.ObjectStorageDataset



The type of osds_object is :red:`osds.utils.ObjectStorageDataset`. Where as :red:`torch.Tensor` is a type of batch object. The batch_size is 20, and number of columns are 24. So the shape of the each batch would be **[20, 24]**.


.. testcode::

    batch = next(iter(DataLoader(taxi_train)))
   
    type(batch)
    >> torch.Tensor

    batch.shape
    >> torch.Size([1, 20, 24])


We also split database based on labels and features. Labels has only one row and 24 columns, so the shape of labels would be **[1, 24]**, each batch of labels will look like one-dimensional tensor.The shape of features for each batch would be **[19, 24]**.

.. testcode::

   labels, features = batch[:, 0], batch[:, 1:]

   labels.shape
   >> torch.Size([1, 24])

   features.shape
   >> torch.Size([1, 19, 24])





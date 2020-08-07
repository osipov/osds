Iterable object from: Google Storage
===========================================

.. raw:: html

    <style> .red {color:#D0312D;  font-size:14px} </style>
    <style> .red {color:#D0312D;  font-size:14px} </style>

.. role:: indexred
.. role:: red

.. toctree::
   :maxdepth: 2


All of the capabilities of :red:`ObjectDataStorage` can also be applied to datasets that reside in serverless object storage services like S3, Google Cloud storage, and Azure. To access datasets from google cloud storage faster and efficient way is to directly call. *f"gcs://*  can be used to call datasets directly from the Google Cloud Storage. 
 
.. testcode::

    from osds.utils import ObjectStorageDataset
    from torch.utils.data import DataLoader


    taxi_train = ObjectStorageDataset(f"gcs://gs://cloud-training-demos/taxifare/large/taxi-train*.csv",  storage_options = {'anon' : False }, batch_size = 2000, worker =4, eager_load_batches=False)

    type(taxi_train)
    >> osds.utils.ObjectStorageDataset



The type of taxi_train object is :red:`osds.utils.ObjectStorageDataset`. Where as :red:`torch.Tensor` is a type of batch object. The batch_size is 10, and number of columns are 7. So the shape of the each batch would be **[10, 7]**.


.. testcode::

    batch = next(iter(DataLoader(taxi_train)))
   
    type(batch)
    >> torch.Tensor

    batch.shape
    >> torch.Size([1, 10, 7])


We also split database based on labels and features. Labels has only one row and 7 columns, so the shape of labels would be **[1, 7]**, each batch of labels will look like one-dimensional tensor.The shape of features for each batch would be **[9, 7]**.

.. testcode::

   labels, features = batch[:, 0], batch[:, 1:]

   labels.shape
   >> torch.Size([1, 7])

   features.shape
   >> torch.Size([1, 9, 7])



To access a dataset from Google cloud storage, another good way is to download dataset to local computer using command line python application gsutils and then create an iterable object using :red:`OSDS` library. Below command will take files from the Google Cloud Storage bucket and copy to a local path on your device. Here in an example, we have downloaded taxi fare public dataset from Google Cloud storage and created an iterable object using  :red:`ObjectDataStorage`. We downloaded the dataset to our local memory, and then createing object so we will start our url from the :red:`file://` intiator. 

.. testcode::

   gsutil cp gs://[BUCKET_NAME]/[OBJECT_NAME] [OBJECT_DESTINATION]

   !gsutil cp gs://cloud-training-demos/taxifare/large/taxi-train*.csv /Path/to/local/folder/



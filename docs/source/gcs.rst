Iterable object from: Google Storage
===========================================

.. raw:: html

    <style> .red {color:#D0312D; font-weight:bold; font-size:16px} </style>
    <style> .red {color:#D0312D; font-weight:bold; font-size:16px} </style>

.. role:: indexred
.. role:: red

.. toctree::
   :maxdepth: 2

To access a dataset from Google cloud storage, the faster and efficient way is to download dataset to local computer using command line python application gsutils and then create an iterable object using :red:`OSDS` library. Below command will take files from the Google Cloud Storage bucket and copy to a local path on your device. 

.. testcode::

   gsutil cp gs://[BUCKET_NAME]/[OBJECT_NAME] [OBJECT_DESTINATION]

Here in an example, we have downloaded taxi fare public dataset from Google Cloud storage and created an iterable object using  :red:`ObjectDataStorage`.

.. testcode::

     !gsutil cp gs://cloud-training-demos/taxifare/large/taxi-train*.csv /Path/to/local/folder/
|
We downloaded the dataset to our local memory, and then createing object so we will start our url from the :red:`file://` intiator. 
|
Let's instantiate the ObjectStorageDatast, and then divide into labels, and features. 
.. testcode::

    from osds.utils import ObjectStorageDataset
    from torch.utils.data import DataLoader


.. testcode::

     taxi_train =  ObjectStorageDataset('file:///content/taxi-train*.csv', 
					 batch_size = 10)

     batch = next(iter(DataLoader(taxi_train)))
     labels, features = batch[:, 0], batch[:, 1:]

The type of taxi_train object is :red:`osds.utils.ObjectStorageDataset`. Where as :red:`torch.Tensor` is a type of batch object. 

.. testcode::

    type(taxi_train)
    >> osds.utils.ObjectStorageDataset
   
    type(batch)
    >> torch.Tensor

let's explore more about labels, and features and batch.The batch_size is 10, and number of columns are 7. So the shape of the each batch would be **[10, 7]**.


.. testcode::

   batch.shape
   >> torch.Size([1, 10, 7])


We also split database based on labels and features. Labels has only one row and 7 columns, so the shape of labels would be **[1, 7]**, each batch of labels will look like one-dimensional tensor, shown below:

.. testcode::

   labels.shape
   >> torch.Size([1, 7])

   labels
   >> tensor([[ 41.0000,   4.0000,   0.0000, -73.9949,  40.7511, -74.0074,  40.7108]],
       dtype=torch.float64)

The shape of features for each batch would be **[9, 7]**. Let's see how one batch of features would look like:


.. testcode::

   features.shape
   >> torch.Size([1, 9, 7])

   features
   >> tensor([[[ 43.5000,   7.0000,   0.0000, -74.0083,  40.7047, -73.8999,  40.8623],
         [ 26.5000,   7.0000,   0.0000, -74.0163,  40.7097, -74.0168,  40.6158],
         [ 22.0000,   4.0000,   0.0000, -73.9727,  40.7560, -73.9408,  40.8332],
         [ 15.7000,   7.0000,   0.0000, -73.9934,  40.7515, -73.9540,  40.7427],
         [ 15.3000,   7.0000,   0.0000, -73.9775,  40.7843, -73.9986,  40.7298],
         [ 21.3000,   4.0000,   0.0000, -73.9767,  40.7650, -73.9692,  40.6932],
         [ 33.0000,   4.0000,   0.0000, -73.9901,  40.7295, -73.9901,  40.7295],
         [ 16.1000,   7.0000,   0.0000, -73.9991,  40.7234, -73.9484,  40.7741],
         [ 42.0000,   4.0000,   0.0000, -73.8713,  40.7736, -73.7310,  40.6749]]],
       dtype=torch.float64)

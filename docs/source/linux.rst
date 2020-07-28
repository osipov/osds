Iterable object from: local file system (Linux)
=========================================================

.. raw:: html

    <style> .red {color:#D0312D; font-weight:bold; font-size:16px} </style>

.. role:: indexred
.. role:: red

.. toctree::
   :maxdepth: 2


Our file is already residing on the local linux file system, so we will start our url from the :red:`file://` intiator. Let's instantiate the :red:`ObjectStorageDatast`, and then divide into labels, and features. 

.. testcode::

    from osds.utils import ObjectStorageDataset
    from torch.utils.data import DataLoader


.. testcode::

     cal_housing =  ObjectStorageDataset("file:///content/sample_data/california_housing_test.csv",) 
					 batch_size = 10)

     batch = next(iter(DataLoader(cal_housing)))
     labels, features = batch[:, 0], batch[:, 1:]

The type of ``cal_housing`` object is :red:`osds.utils.ObjectStorageDataset`. Where as :red:`torch.Tensor` is a type of batch object. 

.. testcode::

    type(cal_housing)
    >> osds.utils.ObjectStorageDataset
   
    type(batch)
    >> torch.Tensor

let's explore more about labels, and features and batch.The batch_size is 10, and number of columns are 9. So the shape of the each batch would be **[10, 9]**.


.. testcode::

   batch.shape
   >> torch.Size([1, 10, 9])


We also split database based on labels and features. Labels has only one row and 9 columns, so the shape of labels would be **[1, 9]**, each batch of labels will look like one-dimensional tensor, shown below:

.. testcode::

   labels.shape
   >> torch.Size([1, 9])

   labels
   >> tensor([[-1.2205e+02,  3.7370e+01,  2.7000e+01,  3.8850e+03,  6.6100e+02,
          1.5370e+03,  6.0600e+02,  6.6085e+00,  3.4470e+05]],
       dtype=torch.float64)

The shape of features of each batch would be **[9, 9]**. Let's see how one batch of features would look like:


.. testcode::

   features.shape
   >> torch.Size([1, 9, 9])

   features
   >> tensor([[[-1.1830e+02,  3.4260e+01,  4.3000e+01,  1.5100e+03,  3.1000e+02,
           8.0900e+02,  2.7700e+02,  3.5990e+00,  1.7650e+05],
         [-1.1781e+02,  3.3780e+01,  2.7000e+01,  3.5890e+03,  5.0700e+02,
           1.4840e+03,  4.9500e+02,  5.7934e+00,  2.7050e+05],
         [-1.1836e+02,  3.3820e+01,  2.8000e+01,  6.7000e+01,  1.5000e+01,
           4.9000e+01,  1.1000e+01,  6.1359e+00,  3.3000e+05],
         [-1.1967e+02,  3.6330e+01,  1.9000e+01,  1.2410e+03,  2.4400e+02,
           8.5000e+02,  2.3700e+02,  2.9375e+00,  8.1700e+04],
         [-1.1956e+02,  3.6510e+01,  3.7000e+01,  1.0180e+03,  2.1300e+02,
           6.6300e+02,  2.0400e+02,  1.6635e+00,  6.7000e+04],
         [-1.2143e+02,  3.8630e+01,  4.3000e+01,  1.0090e+03,  2.2500e+02,
           6.0400e+02,  2.1800e+02,  1.6641e+00,  6.7000e+04],
         [-1.2065e+02,  3.5480e+01,  1.9000e+01,  2.3100e+03,  4.7100e+02,
           1.3410e+03,  4.4100e+02,  3.2250e+00,  1.6690e+05],
         [-1.2284e+02,  3.8400e+01,  1.5000e+01,  3.0800e+03,  6.1700e+02,
           1.4460e+03,  5.9900e+02,  3.6696e+00,  1.9440e+05],
         [-1.1802e+02,  3.4080e+01,  3.1000e+01,  2.4020e+03,  6.3200e+02,
           2.8300e+03,  6.0300e+02,  2.3333e+00,  1.6420e+05]]],
       dtype=torch.float64)

# Licensed under the GNU General Public License v2.0. See footer for details.
import os
import math
import tempfile
import functools

from bisect import bisect_right

import fsspec
import pandas as pd
import numpy as np
import torch as pt

from torch.utils.data import IterableDataset

class ObjectStorageDataset(IterableDataset):
    """
    PyTorch Iterable Dataset with support for a variety of object (and file) stores.

    Wraps `fsspec` to enable support for `s3`, `gcs`, `blob`, `hdfs`, and other storage options.

    Can layer (buffered) text-mode and compression over any file-system, which
    are typically binary-only.

    These instances are safe to serialize, as the low-level file object
    is not created until invoked using `with`.

    Parameters
    ----------
    glob: str, required
        The a protocol handler string specifying a single or multiple files.
        Examples:
        - s3://aws-bucket-name/directory/files*.extension
        - gcs://gcs-bucket-name/folder/file.extension

    storage_options: None or dict, optional
        If not specified or None, then assumed to be {'anon': True} which uses unauthenticated access to object storage. To specify storage platform specific authentication, use {'anon': False} instead. Also, use this dict to specify storage platform specific options, e.g. {'anon': False, 'project': 'myproject'}

    batch_size: None or int, optional
        Number of examples that should be returned per batch for every call to `__iter__` method.

    iterations: None or int, optional
        Number of iterations supported by the `__iter__` method. In case if the `iterations` * `batch_size` is smaller than the number of examples in the dataset, some of the examples from the dataset are never returned by the calls to the `__iter__` method. In case if the `iterations` * `batch_size` is greater than the number of examples in the dataset, some of examples may be returned by the `__iter__` method more than ones.

    dtype: None, str, or dict, optional
        Specification of the data type to use when loading the dataset in memory. When `None`, the widest possible data type is used for numeric data which prevents loss of information but uses extra memory. When using an `str` the same `dtype` is used for all numeric columns in the dataset. When using a Python `dict`, use keys that match the column names (or column indicies) from the source dataset, and values that map to Python native data types or NumPy compatible dtype names. For example: {'col_a': 'Int64', 'col_b': 'int32', 'col_c': 'np.float16'}. Additional information on NumPy dtype specification is available from https://numpy.org/doc/1.18/reference/arrays.dtypes.html. 

    eager_load_batches: None or `Boolean`, optional
        Hint on whether to pre-load partitions from object storage to memory. When not specified, datasets that originate from a local filesystem (glob protocol starts with file://) and fit in memory or cluster memory are pre-loaded to cache. Datasets that do not originate from a local filesystem (for example glob protocol starts with s3:// or gcs://), are not pre-loaded by default unless eager_load_batches is set to True. Avoid setting eager_load_batches to True for datasets that do not fit in the node and cluster memory since this may lead to out of memory conditions.

    fits_in_node_memory: `Boolean`, optional
        Specifies whether the dataset fits in the memory of the node executing this process. When not specified, assumed to be `True`, and if the dataset originates from a local file system, the dataset is pre-loaded to in memory cache. For additional details see `eager_load_batches`.

    fits_in_node_memory: `Boolean`, optional
        Specifies whether the dataset fits in the cluster memory, in other words the total memory available by adding the memory of the individual cluster's nodes. When not specified, assumed to be `True` and the dataset is automatically partitioned based on the number of the `replicas` in the cluster and the `worker` number for the node executing this process such that the datasets partitions are roughly equally distributed across every node in the cluster. Since the nodes in a cluster are not expected to use a local filesystem, unless `eager_load_batches` is specified as `True`, the dataset partitions are not pre-loaded to in memory cache by default

    replicas: `int`, optional
        When `fits_in_cluster_memory` is `True` and `fits_in_node_memory` is `False`, specifies the total number of nodes in the cluster executing this process.

    worker: `int`, optional
        When `fits_in_cluster_memory` is `True` and `fits_in_node_memory` is `False`, specifies the number of the worker in the cluster executing this process. The value must be an integer in the range from 0 (inclusive) to `replicas` (exclusive). The selection of the worker number specifies the partitions of the dataset assigned to this process. For more see `fits_in_cluster_memory`.

    cache_dir: None or str, optional
        Location on the runtime's local file system used to store a local cache of the objects (or files) downloaded from the location specified by `glob`. If `None`, defaults to the platform specific directory returned by `tempfile.gettempdir()`. Otherwise must be a `str` with a valid path on the local filesystem.

    tensor_cache_size: `int`, optional
        Specifies the number of PyTorch tensor instances cached in memory. When not specified, set to `1`, to support the default use case of a dataset that fits in the memory of a node. To avoid out of memory problems, do not specify this cache size unless the batch size is large compared to the size of dataset partitions processed by this node.

    partition_cache_size: `int`, optional
        Specifies the number of dataset partitions (objects from object storage) cached in memory. When not specified or `None`, assumed to be unlimited to support the use case where the entire dataset fits in the memory of the this node.

    batch_cache_size: `int`, optional
        Specifies the number of batches cached in memory. When not specified, set to `1` to support the default use case when the entire dataset fits in the memory of the node and the use case where at most 1 batch fits in the memory of this node.
    """
    def __init__(self, glob, storage_options=None,
                            batch_size=None,
                            dtype = None,
                            iterations=None,
                            eager_load_batches=None,
                            fits_in_node_memory=True,
                            fits_in_cluster_memory=True, replicas=1, worker=0,
                            cache_dir=None, tensor_cache_size=1, partition_cache_size=None, batch_cache_size=1):

        self.glob = glob
        self.dtype = dtype

        # set the platform-specific temporary directory
        cache_dir = cache_dir if cache_dir else tempfile.gettempdir()

        # specify cache allocation for raw tensor data in instances
        self.tensor_cache_size = tensor_cache_size
        if self.tensor_cache_size:
            self.__tensor_by_df_idx = functools.lru_cache(maxsize=self.tensor_cache_size)(self.__tensor_by_df_idx)

        # specify cache allocation for data partitions in instances, None means unlimited
        self.partition_cache_size = partition_cache_size
        if self.partition_cache_size:
            self.__df_by_obj = functools.lru_cache(maxsize=self.partition_cache_size)(self.__df_by_obj)
        else:
            self.__df_by_obj = functools.lru_cache(maxsize=None)(self.__df_by_obj)

        # specify cache allocation per batch in instances
        self.batch_cache_size = batch_cache_size
        if self.batch_cache_size:
            self.__df_by_objs_tuple = functools.lru_cache(maxsize=self.batch_cache_size)(self.__df_by_objs_tuple)

        # find out the protocol of the glob, e.g. s3, gs, hdfs, etc
        protocol, _ = fsspec.core.split_protocol(glob)
        eager_load_batches = True if protocol in ('file') and eager_load_batches is None else eager_load_batches

        # use anonymous connection unless specified otherwise
        storage_options = storage_options if storage_options else {'anon': True}

        # setup a caching filesystem
        self.fs = fsspec.filesystem("filecache",
                                    target_protocol=protocol,
                                    target_options=storage_options,
                                    cache_storage=cache_dir)

        # get the object paths matching the glob
        self.objs = self.fs.glob(glob)
        if not isinstance(self.objs, list) or not len(self.objs):
            raise RuntimeWarning(f"Specified glob pattern {self.glob} failed to match any objects")
        self.objs_indicies = [0]

        self.iterations = iterations if iterations else float('nan')

        if fits_in_node_memory:

            if eager_load_batches:
                self.objs_indicies = self.__expand_obj_idx_in_full(self.objs_indicies, self.objs)
                self.dataset_size = self.__max_batch_idx(self.objs_indicies)
                self.batch_size = batch_size if batch_size else self.dataset_size
            else:
                assert batch_size and type(batch_size) is int and batch_size > 0, "Eager loading batches of batches is disabled, so the batch size must be specified as a positive (greater than 0) integer"
                self.batch_size = batch_size

        elif fits_in_cluster_memory:

            assert type(replicas) is int and replicas > 0, "The number of workers must be a positive integer"
            assert type(worker) is int and worker > -1 and worker < replicas, "The worker must be in the range [0, replicas)"

            self.worker = worker
            self.replicas = replicas

            self.objects_per_worker = int(math.ceil(len(self.objs) / self.replicas))

            self.objs = self.objs[worker * self.objects_per_worker: (worker + 1) * self.objects_per_worker]

            assert batch_size and type(batch_size) is int and batch_size > 0, "The batch size must be specified as a positive (greater than 0) integer"
            self.batch_size = batch_size
            if eager_load_batches:
                self.objs_indicies = self.__expand_obj_idx_in_full(self.objs_indicies, self.objs)

        else:
            assert batch_size and type(batch_size) is int and batch_size > 0, "The batch size must be specified as a positive (greater than 0) integer"
            self.batch_size = batch_size
            if eager_load_batches:
                print("Warning: the batch does not fit in node or in cluster memory but eager loading is enabled (eager_load_batches=True), so you may experience out of memory conditions during eager loading.")
                self.objs_indicies = self.__expand_obj_idx_in_full(self.objs_indicies, self.objs)



    def __obj_idx_by_batch_idx(self, indicies, batch_idx):
        return bisect_right(indicies, batch_idx) - 1

    def __is_obj_in_obj_idx(self, indicies, obj_idx):
        return obj_idx > -1 and obj_idx < (len(indicies) - 1)

    def __max_batch_idx(self, indicies):
        return indicies[-1]

    def __partition_by(self, indicies, spec):
        if max(spec) > self.__max_batch_idx(indicies):
            return [[spec[0] % self.__max_batch_idx(indicies), min(spec[1], self.__max_batch_idx(indicies))],
                    [0, spec[1] % self.__max_batch_idx(indicies)]]
        else:
            return [spec]

    @functools.lru_cache(maxsize=None)
    def __df_by_obj(self, obj):
        df = None
        # print("__df_by_obj ", ps.memory_info())
        with self.fs.open(obj) as file:
            if self.dtype:
                df = pd.read_csv(file, dtype = self.dtype)
            else:
                df = pd.read_csv(file)
        # print("__df_by_obj ", ps.memory_info())
        return df

    @functools.lru_cache(maxsize=1)
    def __df_by_objs_tuple(self, objs):
        dfs = list()
        for obj in objs:
            df = self.__df_by_obj(obj)
            # print("__df_by_obj(", obj, ") id=", id(df))
            dfs.append(df)
        # print("__df_by_objs_tuple ", ps.memory_info())
        df = pd.concat(dfs, copy = False) if len(dfs) > 0 else pd.DataFrame()
        # print("__df_by_objs_tuple ", ps.memory_info())
        return df

    @functools.lru_cache(maxsize=1)
    def __tensor_by_df_idx(self, df_idx_tuple):
        _, df_start_idx, df_end_idx = df_idx_tuple
        # print("__tensor_by_df_idx ", ps.memory_info())
        tensor = pt.tensor(self.df[df_start_idx: df_end_idx].select_dtypes(include=np.number).values)
        # print("__tensor_by_df_idx ", ps.memory_info())
        return tensor

    def __expand_obj_idx_in_full(self, indicies, objs):
        indicies = indicies.copy()

        while not (self.__is_obj_idx_ready(indicies, objs)):

            batch_idx = self.__max_batch_idx(indicies)
            obj_idx = self.__obj_idx_by_batch_idx(indicies, batch_idx)
            df = self.__df_by_obj(objs[obj_idx])
            # print("__df_by_obj(", objs[obj_idx], ") id=", id(df))

            indicies.append(indicies[-1] + len(df))

        return indicies

    def __expand_obj_idx_to_batch_idx(self, indicies, objs, batch_idx):
        indicies = indicies.copy()

        while not (self.__is_obj_idx_ready(indicies, objs)):

            obj_idx = self.__obj_idx_by_batch_idx(indicies, batch_idx)

            if not (self.__is_obj_in_obj_idx(indicies, obj_idx)):

                df = self.__df_by_obj(objs[obj_idx])
                # print("__df_by_obj(", objs[obj_idx], ") id=", id(df))

                indicies.append(indicies[-1] + len(df))

            else:
                break

        return indicies

    def __is_obj_idx_ready(self, indicies, objs):
        return (len(indicies) - 1) == len(objs)

    def __iter__(self):

        batch_start_idx = 0
        batch_end_idx = batch_start_idx + self.batch_size

        while self.iterations:

            if not (self.__is_obj_idx_ready(self.objs_indicies, self.objs)):
                self.objs_indicies = self.__expand_obj_idx_to_batch_idx(self.objs_indicies, self.objs, batch_start_idx)
                self.objs_indicies = self.__expand_obj_idx_to_batch_idx(self.objs_indicies, self.objs, batch_end_idx)

            #failed to expand the object indicies
            if self.objs_indicies == [0]:
                break

            partitions = self.__partition_by(self.objs_indicies, [batch_start_idx, batch_end_idx])
            # print(self.objs_indicies, partitions)

            objs_per_batch = []
            for (start_idx, end_idx) in partitions:
                obj_start_idx = self.__obj_idx_by_batch_idx(self.objs_indicies, start_idx)
                obj_end_idx = self.__obj_idx_by_batch_idx(self.objs_indicies, end_idx)
                objs_per_batch.extend(self.objs[obj_start_idx: obj_end_idx + 1])
            # print(objs_per_batch)

            self.df = self.__df_by_objs_tuple(tuple(objs_per_batch))
            # print("Cachable objs ", objs_per_batch, " id=", id(self.df))

            df_start_idx = batch_start_idx - self.objs_indicies[self.__obj_idx_by_batch_idx(self.objs_indicies, batch_start_idx)]
            df_end_idx = df_start_idx + self.batch_size

            self.tensor = self.__tensor_by_df_idx(tuple([id(self.df), df_start_idx, df_end_idx]))
            # print("Cachable objs ", tuple([id(self.df), df_start_idx, df_end_idx]), " id=", id(self.tensor))
            yield self.tensor

            self.iterations = self.iterations - 1
            if self.__is_obj_idx_ready(self.objs_indicies, self.objs):
                # print(batch_end_idx % self.__max_batch_idx(self.objs_indicies), (batch_end_idx % self.__max_batch_idx(self.objs_indicies)) + self.batch_size)
                batch_start_idx, batch_end_idx = batch_end_idx % self.__max_batch_idx(self.objs_indicies), (batch_end_idx % self.__max_batch_idx(self.objs_indicies)) + self.batch_size
            else:
                # print(batch_end_idx, batch_end_idx + self.batch_size)
                batch_start_idx, batch_end_idx = batch_end_idx, batch_end_idx + self.batch_size

            # print(self.iterations, batch_start_idx, batch_end_idx)

# Copyright 2020 CounterFactual.AI LLC. All Rights Reserved.
#
# Licensed under the GNU General Public License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/osipov/osds/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

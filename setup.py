# Licensed under the GNU General Public License v2.0. See footer for details.
import setuptools

setuptools.setup(
    name="osds",
    version="1.0.42",
    author="CounterFactual.AI",
    author_email="osds@counterfactual.ai",
    description="PyTorch Object Storage Dataset",
    url="http://github.com/osipov/osds",
    license="GPL v2.0",
    install_requires=[
      'fsspec==0.8.3',
      'torch',
      'pandas',
      's3fs==0.4.2',
      'gcsfs==0.6.2'
    ],
    packages=setuptools.find_packages()
)
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

# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""EDIS dataset processing script."""
import pickle
import numpy as np

from mindspore.dataset import GeneratorDataset


class GNNDataset:
    """Class for Generate Dataset."""

    def __init__(self, fpath, train=True, config=None):
        self.batch_size = config.test.batch_size
        if train:
            self.train_data = pickle.load(open(fpath + '/data_train.pkl', 'rb'))
        else:
            self.train_data = pickle.load(open(fpath + '/data_test.pkl', 'rb'))
        self.train_index = None
        self.column_name = ["x_feature", "x_mask", "edge_attr", "edge_feature",
                            "edge_mask", "target_feature", "target_mask",
                            "label", "batch_info", "index_all"]

    def __getitem__(self, index):
        index_all = self.train_index[index * self.batch_size:(index + 1) * self.batch_size]
        train_feature = self.process_data(self.train_data, self.batch_size, index_all)

        x_feature = train_feature.get('x_feature_batch')
        x_mask = train_feature.get('x_mask_batch')
        edge_attr = train_feature.get('edge_attr_batch')
        edge_feature = train_feature.get('edge_feature_batch')
        edge_mask = train_feature.get('edge_mask_batch')
        target_feature = train_feature.get('target_feature_batch')
        target_mask = train_feature.get('target_mask_batch')
        label = train_feature.get('label_batch')
        batch_info = train_feature.get('batch_info')
        res = x_feature, x_mask, edge_attr, edge_feature, edge_mask, \
              target_feature, target_mask, label, batch_info, index_all

        return res

    def __len__(self):
        return int(len(self.train_index) / self.batch_size)

    @staticmethod
    def process_data(train_loader, batch_size, index_all):
        """process data"""
        max_edge = 320
        max_node = 144

        x_feature_batch = []
        edge_feature_batch = []
        edge_attr_batch = []
        label_batch = []
        batch_info = []
        target_feature_batch = []
        node_num_all = 0
        edge_num_all = 0

        for i_num, index in enumerate(index_all):

            data = train_loader[index]
            edge_index = data["edge_index"]
            edge_attr = data["edge_attr"]
            target = [data["target"].tolist()]
            x = data["x"].tolist()
            label = data["y"].tolist()
            edge_num = len(edge_index[0])
            node_num = len(x)
            batch = [i_num] * node_num
            batch_edge = [i_num] * edge_num

            if i_num == 0:
                x_feature_batch = x
                edge_feature_batch = edge_index.tolist()
                edge_attr_batch = edge_attr.tolist()
                target_feature_batch = target
                batch_info = batch
                batch_info_edge = batch_edge
                label_batch = label
                node_num_all = node_num
                edge_num_all = edge_num

            else:
                x_feature_batch.extend(x)
                edge_feature_batch[0].extend(list(edge_index[0] + node_num_all))
                edge_feature_batch[1].extend(list(edge_index[1] + node_num_all))
                edge_attr_batch.extend(edge_attr.tolist())
                target_feature_batch.extend(target)
                batch_info.extend(batch)
                batch_info_edge.extend(batch_edge)
                label_batch.extend(label)
                node_num_all += node_num
                edge_num_all += edge_num

        x_feature_batch = np.array(x_feature_batch)
        edge_feature_batch = np.array(edge_feature_batch)
        edge_attr_batch = np.array(edge_attr_batch)
        target_feature_batch = np.array(target_feature_batch)
        batch_info = np.array(batch_info)
        batch_info_edge = np.array(batch_info_edge)
        label_batch = np.array(label_batch)

        x_1 = np.zeros((max_node * batch_size, 22))
        x_mask1 = np.zeros((max_node * batch_size,))
        target_feat1 = target_feature_batch
        edge_feat1 = np.ones((2, max_edge * batch_size)) * node_num_all
        edge_mask1 = np.zeros((2, max_edge * batch_size))
        edge_attr1 = np.zeros((max_edge * batch_size, 6))
        label1 = label_batch
        batch1 = np.zeros((max_node * batch_size,)).tolist()
        batch2 = np.zeros((max_edge * batch_size,)).tolist()

        x_1[:node_num_all] = x_feature_batch
        batch1[:node_num_all] = batch_info[:]
        batch1[node_num_all:] = [len(index_all)] * (max_node * batch_size - node_num_all)
        batch1 = np.array(batch1)
        batch2[:edge_num_all] = batch_info_edge[:]
        batch2[edge_num_all:] = np.array([len(index_all)] * (max_edge * batch_size - edge_num_all))
        batch2 = np.array(batch2)
        x_mask1[:node_num_all] = 1
        edge_feat1[0][:edge_num_all] = edge_feature_batch[0]
        edge_feat1[1][:edge_num_all] = edge_feature_batch[1]
        edge_mask1[0][:edge_num_all] = 1
        edge_mask1[1][:edge_num_all] = 1
        edge_attr1[:edge_num_all] = edge_attr_batch

        new_train_data = {"x_feature_batch": x_1.astype(np.float32),
                          "x_mask_batch": x_mask1.astype(np.int32),
                          "edge_attr_batch": edge_attr1.astype(np.float32),
                          "edge_feature_batch": edge_feat1.astype(np.int32),
                          "edge_mask_batch": edge_mask1.astype(np.int32),
                          "target_feature_batch": target_feat1.astype(np.int64),
                          "target_mask_batch": np.zeros((batch_size, 1200)).astype(np.int32),
                          "label_batch": label1.astype(np.float32),
                          "batch_info": batch1.astype(np.int32),
                          "batch_info_edge": batch2.astype(np.int32),
                          }
        return new_train_data

    def create_iterator(self, num_epochs=1):
        """create data iterator"""
        index_all = list(range(len(self.train_data)))
        self.train_index = []
        for _ in range(num_epochs):
            self.train_index.extend(index_all)

        dataset = GeneratorDataset(source=self, column_names=self.column_name,
                                   num_parallel_workers=4, shuffle=True, max_rowsize=16)
        iteration = dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)
        return iteration

    def process(self):
        index_all = [0]
        # for inference, only one data at a time
        feature_dict = self.process_data(self.train_data, 1, index_all)
        return feature_dict

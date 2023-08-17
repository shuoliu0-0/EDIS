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
"""test"""
import os
import argparse
import copy
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor
from mindspore import context
from mindspore.common import mutable

from utils import AverageMeter, get_cindex, get_rm2, load_config
from dataset.Dataset import GNNDataset
from model import MGraphDTA, EDISMOE
from log.train_logger import Logger


def get_acc_ensemble(ensemble, criterion, test_loader):
    """ensemble"""
    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_r2 = AverageMeter()

    for data in test_loader:
        preds = []
        inputs_feats = [np.array(data["x_feature"], np.float32), \
                        np.array(data["edge_attr"], np.float32), \
                        np.array(data["edge_feature"], np.int32), \
                        np.array(data["target_feature"], np.float32), \
                        np.array(data["batch_info"], np.int32), \
                        np.array(data["x_mask"], np.float32),]
        inputs_feat = [Tensor(feat) for feat in inputs_feats]
        inputs_feat = mutable(inputs_feat)
        for i, model in enumerate(ensemble):
            if i == 0:
                inputs_feat1 = copy.deepcopy(inputs_feat)
                pred, f = model(*inputs_feat1)
            else:
                pred, _ = model(inputs_feat, f)
            preds.append(pred)

        pred_prob = ops.stack(preds, axis=0).mean(axis=0)
        pred_prob = pred_prob.asnumpy().reshape(-1)

        loss = criterion(Tensor(pred_prob, mindspore.float32),
                         Tensor(np.array(data["label"], np.float32).reshape(-1), mindspore.float32))
        cindex = get_cindex(np.array(data["label"], np.float32).reshape(-1), pred_prob)
        r2 = get_rm2(np.array(data["label"], np.float32).reshape(-1), pred_prob)

        running_loss.update(loss.asnumpy(), data["label"].shape[0])
        running_cindex.update(cindex, data["label"].shape[0])
        running_r2.update(r2, data["label"].shape[0])

    epoch_loss = running_loss.get_average()
    epoch_cindex = running_cindex.get_average()
    epoch_r2 = running_r2.get_average()
    running_loss.reset()
    running_cindex.reset()
    running_r2.reset()

    return epoch_loss, epoch_cindex, epoch_r2


def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--config', default='./EDIS_config.yaml', help='configuration for test')
    parser.add_argument('--dataset', default='davis', help='davis or kiba')

    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    
    data_root = os.path.join(config.data.data_root, args.dataset)
    pretrain_models = config.test.pretrain_models
    pretrain_moe = config.test.pretrain_moe
    
    params = dict(
        data_root=data_root,
        save_dir=f"{data_root}/save/",
        dataset=args.dataset,
        batch_size=config.test.batch_size
    )

    logger = Logger(params)
    logger.info(__file__)

    context.set_context(device_id=config.test.device_id)

    test_set = GNNDataset(data_root, train=False, config=config)
    test_loader = test_set.create_iterator(num_epochs=1)

    ensemble = [MGraphDTA(config.test.block_num, config.test.embedding_size,
                          config.test.filter_num, config.test.out_dim),
                EDISMOE(config.test.filter_num * 3 * 2, config.test.num_experts,
                        config.test.hidden_size, config.test.noisy_gating,
                        config.test.k, config.test.block_num,
                        config.test.embedding_size, config.test.filter_num,
                        config.test.out_dim)]

    
    criterion = nn.MSELoss()

    param_dict = mindspore.load_checkpoint(pretrain_models)
    mindspore.load_param_into_net(ensemble[0], param_dict)
    param_dict = mindspore.load_checkpoint(pretrain_moe)
    mindspore.load_param_into_net(ensemble[1], param_dict)

    ensemble_loss, ensemble_cindex, ensemble_r2 = get_acc_ensemble(ensemble, criterion, test_loader)
    msg0 = "ensemble_MSE-%.4f" % (ensemble_loss)
    logger.info(msg0)


if __name__ == "__main__":
    main()

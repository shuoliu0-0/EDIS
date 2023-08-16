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
"""train logger"""
import os
import sys
import time
from log.basic_logger import BasicLogger

if sys.path[-1] != os.getcwd():
    sys.path.append(os.getcwd())


def create_dir(dir_list):
    assert isinstance(dir_list, list) is True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)


class Logger(BasicLogger):
    """logger"""
    def __init__(self, args):
        self.args = args

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.get('mark') is None:
            savetag = timestamp + '_' + args.get('dataset')

        save_dir = args.get('save_dir')
        if save_dir is None:
            raise Exception('save_dir can not be None!')
        train_save_dir = os.path.join(save_dir, savetag)
        self.log_dir = os.path.join(train_save_dir, 'log', 'train')
        self.model_dir = os.path.join(train_save_dir, 'model')
        create_dir([self.log_dir, self.model_dir])

        print(self.log_dir)

        log_path = os.path.join(self.log_dir, 'Train.log')
        super().__init__(log_path)

    def get_log_dir(self):
        if hasattr(self, 'log_dir'):
            return self.log_dir
        return None

    def get_model_dir(self):
        if hasattr(self, 'model_dir'):
            return self.model_dir
        return None
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
"""basic logger"""
import logging


class BasicLogger():
    """basic logger"""
    def __init__(self, path):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        if not self.logger.handlers:
            # Create a file handler
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            # use StreamHandler for print
            print_handler = logging.StreamHandler()
            print_handler.setLevel(logging.DEBUG)
            print_handler.setFormatter(formatter)

            # Add the handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(print_handler)

    def noteset(self, message):
        self.logger.noteset(message)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == "__main__":
    logger = BasicLogger('test.log')
    logger.info("This is a test")
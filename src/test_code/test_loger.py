from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
import tensorboardX

class Logger(object):
    def __init__(self):
        print('Start Logging')
        os.makedirs('./test_log')
        self.writer = tensorboardX.SummaryWriter(log_dir='test_log')

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)


logger = Logger()
for i in range(1000):
    logger.scalar_summary('train', i, i)


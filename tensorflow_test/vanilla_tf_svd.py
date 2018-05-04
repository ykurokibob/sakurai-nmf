from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices



def main(_):
    num_iterts = 10
    _size = [1000, 1000]
    inputs = tf.placeholder(tf.float32, shape=_size, name='inputs')
    _inputs = np.random.uniform(0., 1., size=_size)
    
    # with tf.device('/cpu:0'):
    suv = tf.svd(inputs, compute_uv=True)
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        init.run()
        
        start_time = time.time()
        
        for _ in range(num_iterts):
            sess.run(suv, feed_dict={inputs: _inputs})
        
        end_time = time.time()
    
    duration = (end_time - start_time) / num_iterts
    print('Duration of computing svd: {}.'.format(duration))


if __name__ == '__main__':
    np.random.seed(42)
    tf.set_random_seed(42)
    print(list_local_devices())
    tf.app.run()

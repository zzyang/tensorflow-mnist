# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:43:05 2018

@author: andyan01
"""

import tensorflow as tf

hello = tf.constant("Hello TensorFlow!")

sess = tf.Session()

print(sess.run(hello))
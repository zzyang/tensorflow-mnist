# -*- coding: utf-8 -*-

from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#import cv2

def imageprepare():

    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open('./dst.png').convert('F')

    #im.save("./sample.png")
    plt.imshow(im)
    plt.show()
    #tv = list(im.getdata()) #get pixel values
    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    #tva = [ (255-x)*1.0/255.0 for x in tv] 
    
    tv = np.array(im)
    tva = (255-tv)*1.0/255.0
    
    #print(tva)
    return tva


    # Define the model (same as when creating the model file)
result=imageprepare()

X = tf.placeholder(tf.float32, [None, 28, 28, 1])


K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
#YY4 = tf.nn.dropout(Y4, pkeep)
#Ylogits = tf.matmul(YY4, W5) + B5
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)


init = tf.global_variables_initializer()
#saver = tf.train.Saver()  # defaults to saving all variables

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph('./Model/model.ckpt.meta')
    
    saver.restore(sess, './Model/model.ckpt')

    print ("Model restored.")

    prediction=tf.argmax(Y,1)
    r = tf.reshape(result, [-1, 28, 28, 1]).eval()
    
    predint=sess.run(prediction, feed_dict={X: r})

    print('recognize result:')
    print(predint[0])
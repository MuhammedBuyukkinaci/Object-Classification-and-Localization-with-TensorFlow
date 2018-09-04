import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd

MAIN_DIR = os.getcwd()
TEST_DIR = os.path.join(os.getcwd(),"testing_images")

os.chdir(TEST_DIR)

#HyperParameters
epochs = 5
step_size = 4
IMG_SIZE_ALEXNET = 227 # image size
validating_size = 89 # while cross validating, we are evaluating batch by batch
nodes_fc1 = 4096 # no of nodes on fc layer 1
nodes_fc2 = 4096 # no of nodes on fc layer 2
output_classes = 3 # three classes: eggplant, 
output_locations = 4 # minx, miny, maxx, maxy

def make_test_data():
    testing_photos = []
    for i in os.listdir():
        print(i)
        path = os.path.join(os.getcwd(),i)
        img = cv2.imread(path,1)
        img = cv2.resize(img,(IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET))
        testing_photos.append(img)
    return np.array(testing_photos)

test_photos = make_test_data()

os.chdir(MAIN_DIR)

#saver = tf.train.Saver()
#new_saver = tf.train.import_meta_graph('CNN_OL.ckpt.meta')


#Resetting graph
tf.reset_default_graph()

#Defining Placeholders
x = tf.placeholder(tf.float32,shape=[None,IMG_SIZE_ALEXNET,IMG_SIZE_ALEXNET,3])
y_true_1 = tf.placeholder(tf.float32,shape=[None,output_classes])
y_true_2 = tf.placeholder(tf.float32,shape=[None,output_locations])

##CONVOLUTION LAYER 1
#Weights for layer 1
w_1 = tf.Variable(tf.truncated_normal([11,11,3,96], stddev=0.01))
#Bias for layer 1
b_1 = tf.Variable(tf.constant(0.0, shape=[[11,11,3,96][3]]))
#Applying convolution
c_1 = tf.nn.conv2d(x, w_1,strides=[1, 4, 4, 1], padding='VALID')
#Adding bias
c_1 = c_1 + b_1
#Applying RELU
c_1 = tf.nn.relu(c_1)
								
print(c_1)
##POOLING LAYER1
p_1 = tf.nn.max_pool(c_1, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
print(p_1)

##CONVOLUTION LAYER 2
#Weights for layer 2
w_2 = tf.Variable(tf.truncated_normal([5,5,96,256], stddev=0.01))
#Bias for layer 2
b_2 = tf.Variable(tf.constant(1.0, shape=[[5,5,96,256][3]]))
#Applying convolution
c_2 = tf.nn.conv2d(p_1, w_2,strides=[1, 1, 1, 1], padding='SAME')
#Adding bias
c_2 = c_2 + b_2
#Applying RELU
c_2 = tf.nn.relu(c_2)

print(c_2)

##POOLING LAYER2
p_2 = tf.nn.max_pool(c_2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
print(p_2)

##CONVOLUTION LAYER 3
#Weights for layer 3
w_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01))
#Bias for layer 3
b_3 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 256, 384][3]]))
#Applying convolution
c_3 = tf.nn.conv2d(p_2, w_3,strides=[1, 1, 1, 1], padding='SAME')
#Adding bias
c_3 = c_3 + b_3
#Applying RELU
c_3 = tf.nn.relu(c_3)

print(c_3)

##CONVOLUTION LAYER 4
#Weights for layer 4
w_4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01))
#Bias for layer 4
b_4 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 384][3]]))
#Applying convolution
c_4 = tf.nn.conv2d(c_3, w_4,strides=[1, 1, 1, 1], padding='SAME')
#Adding bias
c_4 = c_4 + b_4
#Applying RELU
c_4 = tf.nn.relu(c_4)

print(c_4)

##CONVOLUTION LAYER 5
#Weights for layer 5
w_5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01))
#Bias for layer 5
b_5 = tf.Variable(tf.constant(0.0, shape=[[3, 3, 384, 256][3]]))
#Applying convolution
c_5 = tf.nn.conv2d(c_4, w_5,strides=[1, 1, 1, 1], padding='SAME')
#Adding bias
c_5 = c_5 + b_5
#Applying RELU
c_5 = tf.nn.relu(c_5)

print(c_5)

##POOLING LAYER3
p_3 = tf.nn.max_pool(c_5, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')
print(p_3)

#Flattening
flattened = tf.reshape(p_3,[-1,6*6*256])
print(flattened)

##Fully Connected Layer 1
#Getting input nodes in FC layer 1
input_size = int( flattened.get_shape()[1] )
#Weights for FC Layer 1
w1_fc = tf.Variable(tf.truncated_normal([input_size, nodes_fc1], stddev=0.01))
#Bias for FC Layer 1
b1_fc = tf.Variable( tf.constant(1.0, shape=[nodes_fc1] ) )
#Summing Matrix calculations and bias
s_fc1 = tf.matmul(flattened, w1_fc) + b1_fc
#Applying RELU
s_fc1 = tf.nn.relu(s_fc1)

#Dropout Layer 1
hold_prob1 = tf.placeholder(tf.float32)
s_fc1 = tf.nn.dropout(s_fc1,keep_prob=hold_prob1)

print(s_fc1)

##Fully Connected Layer 2
#Weights for FC Layer 2
w2_fc = tf.Variable(tf.truncated_normal([nodes_fc1, nodes_fc2], stddev=0.01))
#Bias for FC Layer 2
b2_fc = tf.Variable( tf.constant(1.0, shape=[nodes_fc2] ) )
#Summing Matrix calculations and bias
s_fc2 = tf.matmul(s_fc1, w2_fc) + b2_fc
#Applying RELU
s_fc2 = tf.nn.relu(s_fc2)
print(s_fc2)

#Dropout Layer 2
hold_prob2 = tf.placeholder(tf.float32)
s_fc2 = tf.nn.dropout(s_fc2,keep_prob=hold_prob1)

##Fully Connected Layer 3 -- CLASSIFICATION HEAD
#Weights for FC Layer 3
w3_fc_1 = tf.Variable(tf.truncated_normal([nodes_fc2,output_classes], stddev=0.01))
#Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
b3_fc_1 = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
#Summing Matrix calculations and bias
y_pred_1 = tf.matmul(s_fc2, w3_fc_1) + b3_fc_1
#Applying RELU
print(y_pred_1)

##Fully Connected Layer 3 -- REGRESSION HEAD
#Weights for FC Layer 3
w3_fc_2 = tf.Variable(tf.truncated_normal([nodes_fc2,output_locations], stddev=0.01))
#Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
b3_fc_2 = tf.Variable( tf.constant(1.0, shape=[output_locations] ) )
#Summing Matrix calculations and bias
y_pred_2 = tf.matmul(s_fc2, w3_fc_2) + b3_fc_2
#Applying RELU
print(y_pred_2)

with tf.Session() as session:
    new_saver = tf.train.Saver()
    new_saver.restore(session, "CNN_OL.ckpt")
    print("Model restored.") 
    print('Initialized')
    k,l = session.run([tf.nn.softmax(y_pred_1),y_pred_2], feed_dict={x:test_photos , hold_prob1:1,hold_prob2:1})


pred_labels = []
for i in range(len(k)):
    r = np.round(k[i],3).argmax()
    if r ==0 : pred_labels.append("cucumber")
    elif r ==1: pred_labels.append("eggplant")
    elif r ==2: pred_labels.append("mushroom")	
	
#Multiple images parameters
w=227
h=227
columns = 4
rows = 4


fig = plt.figure(figsize=(20, 20))
for m in range(1, columns*rows +1):
    img = test_photos[m-1].reshape([IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET, 3])
    fig.add_subplot(rows, columns, m)
    if pred_labels[m-1] =='cucumber':
        plt.imshow(img)
        cv2.rectangle(img, (l[m-1][0], l[m-1][1]), (l[m-1][2], l[m-1][3]), (0,255,0), 2)
    elif pred_labels[m-1] =='eggplant':
        plt.imshow(img)
        cv2.rectangle(img, (l[m-1][0], l[m-1][1]), (l[m-1][2], l[m-1][3]), (255,0,0), 2)
    else:
        plt.imshow(img)
        cv2.rectangle(img, (l[m-1][0], l[m-1][1]), (l[m-1][2], l[m-1][3]), (0,0,255), 2)
    plt.title("Pred: " + pred_labels[m-1])
    plt.axis('off')
plt.show()


fig = plt.figure(figsize=(20, 20))
for m in range(1, columns*rows +1):
    img = test_photos[m-1].reshape([IMG_SIZE_ALEXNET, IMG_SIZE_ALEXNET, 3])
    fig.add_subplot(rows, columns, m)
    if pred_labels[m-1] =='cucumber':
        plt.imshow(img)
        cv2.rectangle(img, (l[m-1][0], l[m-1][1]), (l[m-1][2], l[m-1][3]), (0,255,0), 2)
    elif pred_labels[m-1] =='eggplant':
        plt.imshow(img)
        cv2.rectangle(img, (l[m-1][0], l[m-1][1]), (l[m-1][2], l[m-1][3]), (255,0,0), 2)
    else:
        plt.imshow(img)
        cv2.rectangle(img, (l[m-1][0], l[m-1][1]), (l[m-1][2], l[m-1][3]), (0,0,255), 2)
    plt.title("Pred: " + pred_labels[m-1])
    plt.axis('off')
plt.show()	

print("Testing finished")
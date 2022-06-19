########################################
import numpy as np
import time
# import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import *
from keras.callbacks import ModelCheckpoint


########################################
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


########################################
''' 
期末作業進行步驟：
1: read train_set.json
2: 依train_set.json 獲得所有 training data pair(image, target_dict)
3: resize all iamge
4: training
5: test (以此數據進行評分)


提示：
1: model input: resized image in train_set (test_set同樣做法)
2: model output: resized plate_dict['車牌角點'] in train_set (test_set同樣做法)
3: model 不是分類問題，而是regression問題 (與作業 autoEncoder 類似)

'''

import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time
#%matplotlib inline

#ROOT_DIR = os.getcwd()
ROOT_DIR = './'

# input path
Dataset_Dir = os.path.join(ROOT_DIR, 'car_dataset')


# read train/test set
name = os.path.join(ROOT_DIR, 'train_set.json')
with open(name, 'r') as file:
    train_files = json.load(file)
    
name = os.path.join(ROOT_DIR, 'test_set.json')
with open(name, 'r') as file:
    test_files = json.load(file)
    
print('train_files:', len(train_files), train_files[0])
print('test_files:', len(test_files), test_files[0])


########################################
# test: read pair: .jpg & .json
def get_pair_info(name):
    f1 = name + '.json'
    with open(f1, 'r') as file:
        jdict = json.load(file)
        
    f2 = name + '.jpg'
    img = cv2.imdecode( np.fromfile(f2, dtype=np.uint8), -1)
    return img, jdict

# 範例
case_file = train_files[0] # 第 1 筆資料
name = os.path.join(Dataset_Dir, case_file)
src_img, plate_dict = get_pair_info(name)

print('src_img:', src_img.shape)
print('plate_dict:', plate_dict)


########################################
Red_color = (0, 0, 255) # BGR format
Green_color = (0, 200, 0)
radius = 2

draw_img = src_img.copy()
x1,y1, x2,y2 = plate_dict['車牌位置']
px1,py1, px2,py2, px3,py3, px4,py4 = plate_dict['車牌角點']

cv2.rectangle(draw_img, (x1,y1), (x2,y2), Green_color, 1)
cv2.circle(draw_img, (px1,py1),  radius, Red_color, -1)
cv2.circle(draw_img, (px2,py2),  radius, Red_color, -1)
cv2.circle(draw_img, (px3,py3),  radius, Red_color, -1)
cv2.circle(draw_img, (px4,py4),  radius, Red_color, -1)

plt.figure(figsize=(20,10))
plt.subplot(121); plt.imshow(src_img[:,:,::-1])
plt.subplot(122); plt.imshow(draw_img[:,:,::-1])


########################################
# 範例
case_file = train_files[20] # 第 1 筆資料
name = os.path.join(Dataset_Dir, case_file)
src_img, plate_dict = get_pair_info(name)

print('src_img:', src_img.shape)
print('plate_dict:', plate_dict)

Red_color = (0, 0, 255) # BGR format
Green_color = (0, 200, 0)
radius = 2

draw_img = src_img.copy()
x1,y1, x2,y2 = plate_dict['車牌位置']
px1,py1, px2,py2, px3,py3, px4,py4 = plate_dict['車牌角點']

cv2.rectangle(draw_img, (x1,y1), (x2,y2), Green_color, 1)
cv2.circle(draw_img, (px1,py1),  radius, Red_color, -1)
cv2.circle(draw_img, (px2,py2),  radius, Red_color, -1)
cv2.circle(draw_img, (px3,py3),  radius, Red_color, -1)
cv2.circle(draw_img, (px4,py4),  radius, Red_color, -1)

plt.figure(figsize=(20,10))
plt.subplot(121); plt.imshow(src_img[:,:,::-1])
plt.subplot(122); plt.imshow(draw_img[:,:,::-1])


########################################
def img_resize(image,size_x,size_y):
    data_image=cv2.resize(image,(size_x,size_y),
                          interpolation=cv2.INTER_AREA)
    data_image=data_image/255
    return data_image

def label_resize(size_x,size_y,px1,py1, px2,py2, px3,py3, px4,py4,src_img):
    x1=round(px1*(size_x/src_img.shape[1]))
    y1=round(py1*(size_y/src_img.shape[0]))
    x2=round(px2*(size_x/src_img.shape[1]))
    y2=round(py2*(size_y/src_img.shape[0]))
    x3=round(px3*(size_x/src_img.shape[1]))
    y3=round(py3*(size_y/src_img.shape[0]))
    x4=round(px4*(size_x/src_img.shape[1]))
    y4=round(py4*(size_y/src_img.shape[0]))
    return [x1,y1,x2,y2,x3,y3,x4,y4]

def draw(data_image,x1,y1,x2,y2,x3,y3,x4,y4):
    Red_color = (0, 0, 255)
    radius = 2
    cv2.circle(data_image, (x1,y1),  radius, Red_color, -1)
    cv2.circle(data_image, (x2,y2),  radius, Red_color, -1)
    cv2.circle(data_image, (x3,y3),  radius, Red_color, -1)
    cv2.circle(data_image, (x4,y4),  radius, Red_color, -1)
    plt.figure(figsize=(20,10))
    plt.subplot(121); plt.imshow(data_image[:,:,::-1])

def draw_2(data_image,label):
    Red_color = (0, 0, 255)
    radius = 2
    cv2.circle(data_image, (label[0],label[1]),  radius, Red_color, -1)
    cv2.circle(data_image, (label[2],label[3]),  radius, Red_color, -1)
    cv2.circle(data_image, (label[4],label[5]),  radius, Red_color, -1)
    cv2.circle(data_image, (label[6],label[7]),  radius, Red_color, -1)
    plt.figure(figsize=(20,10))
    plt.subplot(121); plt.imshow(data_image[:,:,::-1])

def draw_3(data_image,label,test_label):
    Red_color = (0, 0, 255)
    Green_color = (0, 200, 0)
    radius = 2
    cv2.circle(data_image, (label[0],label[1]),  radius, Red_color, -1)
    cv2.circle(data_image, (label[2],label[3]),  radius, Red_color, -1)
    cv2.circle(data_image, (label[4],label[5]),  radius, Red_color, -1)
    cv2.circle(data_image, (label[6],label[7]),  radius, Red_color, -1)
    cv2.circle(data_image, (test_label[0],test_label[1]),  radius, Green_color, -1)
    cv2.circle(data_image, (test_label[2],test_label[3]),  radius, Green_color, -1)
    cv2.circle(data_image, (test_label[4],test_label[5]),  radius, Green_color, -1)
    cv2.circle(data_image, (test_label[6],test_label[7]),  radius, Green_color, -1)
    plt.figure(figsize=(20,10))
    plt.subplot(121); plt.imshow(data_image[:,:,::-1])


########################################
# image size
size_x, size_y = 256, 256

train_data=np.zeros((len(train_files),size_x,size_y,3),dtype=float)
train_label=np.zeros((len(train_files),8),dtype=int)

test_data=np.zeros((len(test_files),size_x,size_y,3),dtype=float)
test_label=np.zeros((len(test_files),8),dtype=int)

for i in range(len(test_files)):
    case_file = test_files[i]
    name = os.path.join(Dataset_Dir, case_file)
    src_img, plate_dict = get_pair_info(name)
    test_data_image=img_resize(src_img,size_x,size_y)
    px1,py1, px2,py2, px3,py3, px4,py4 = plate_dict['車牌角點']
    test_data_label=label_resize(size_x,size_y,px1,py1, px2,py2, px3,py3, px4,py4,src_img)
    test_label[i]=test_data_label
    test_data[i]=test_data_image

for i in range(len(train_files)):
    case_file = train_files[i]
    name = os.path.join(Dataset_Dir, case_file)
    src_img, plate_dict = get_pair_info(name)
    train_data_image=img_resize(src_img,size_x,size_y)
    px1,py1, px2,py2, px3,py3, px4,py4 = plate_dict['車牌角點']
    train_data_label=label_resize(size_x,size_y,px1,py1, px2,py2, px3,py3, px4,py4,src_img)
    train_label[i]=train_data_label
    train_data[i]=train_data_image

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)


########################################
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=11)


########################################
model = tf.keras.Sequential()

#conv1
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], input_shape=(size_x,size_y, 3), strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
#conv2
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#conv3
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv4
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



#conv5
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv6
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv7
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#conv8
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv9
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv10
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


#conv11
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv12
model.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.Dropout(0.4))
#conv13
model.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=[3, 3], strides=1,activation='relu',
                                        padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization()) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.5))


model.add(tf.keras.layers.Dense(units=512,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
model.add(tf.keras.layers.BatchNormalization())


model.add(tf.keras.layers.Dense(units=8))
model.summary()


########################################
print('======== training ========')
batch_size=8
#adam = Adam(learning_rate=0.01)
sgd = SGD(lr=1e-2, clipnorm=1.)    #1e-3 
model.compile(loss='mean_squared_error', optimizer=sgd)
epochs=10
filepath="./model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
mode='min')
callbacks_list = [checkpoint]
train_history = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs, verbose=1,
                        validation_data=(X_val, y_val),callbacks=callbacks_list)


########################################
output = model.predict(test_data)


########################################
print('======== loss ========')
loss = model.evaluate(test_data, test_label)
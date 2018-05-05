import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential,Model,load_model
from keras.layers import *
from keras.optimizers import Adam
import load_pictures
import sys

valid_dir = "hw3-train-validation/validation/"
predict_dir = "hw3-train-validation/prediction/"
valid_size = 257
model = Sequential()
model_name = "model.h5"


def build_model():
    #block 1
    model.add(Conv2D(name='block1_conv1',input_shape=(512,512,3),filters=64,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block1_conv2',filters=64,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    #block 2
    model.add(Conv2D(name='block2_conv1',filters=128,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block2_conv2',filters=128,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    #block 3
    model.add(Conv2D(name='block3_conv1',filters=256,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block3_conv2',filters=256,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block3_conv3',filters=256,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    #block 4
    model.add(Conv2D(name='block4_conv1',filters=512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block4_conv2',filters=512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block4_conv3',filters=512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    #block 5
    model.add(Conv2D(name='block5_conv1',filters=512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block5_conv2',filters=512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(name='block5_conv3',filters=512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
    #FCN32
    model.add(Conv2D(filters=4096,kernel_size=(2,2),padding='same',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=4096,kernel_size=(1,1),padding='same',activation='relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=7,kernel_size=(1,1),padding='valid',activation='linear',kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(filters=7,kernel_size=64,strides=32,padding='same',activation='softmax'))
    
    print("* Model is Builded ")


idx = []
for i in range(valid_size):
    if i<10 : idx.append('000'+str(i))
    elif 10<=i<100: idx.append('00'+str(i))
    elif 100<=i<1000: idx.append('0'+str(i))
    else : idx.append(str(i))


def main():
    print("======= main ========")
    build_model()
    model.load_weights(model_name)
    valid_x = np.load("valid_x.npy")
    result = model.predict(valid_x)
    for i in range(valid_size):
        plt.imsave(predict_dir+idx[i]+"_mask.png",reconstruct(result[i]))

def reconstruct(predict): # (,,7)to (,,3)
    color = np.empty((512,512))
    rgb = np.empty((512,512,3),dtype=np.float32)
    for i in range(512):
        for j in range(512):
            color[i][j] = np.argmax(predict[i][j])
    rgb[ color == 0 ] = [1,1,1]
    rgb[ color == 1 ] = [1,1,0]
    rgb[ color == 2 ] = [1,0,1]
    rgb[ color == 3 ] = [0,1,1]
    rgb[ color == 4 ] = [0,1,0]
    rgb[ color == 5 ] = [0,0,1]
    rgb[ color == 6 ] = [0,0,0]
    return rgb


if __name__ == "__main__":
    main()
    
import numpy as np
from matplotlib import pyplot as plt
import  matplotlib.image as mpimg
from keras.utils import np_utils
from keras.models import Sequential,Model,load_model
from keras.layers import *
from keras.optimizers import Adam
import sys
import os

test_dir = sys.argv[1]
predict_dir = sys.argv[2]
model = Sequential()
model_name = "model.h5"

img_list = [ img for img in os.listdir(test_dir) if img.endswith(".jpg")]
img_list.sort()


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

def main():
    print("======= main ========")
    build_model()
    model.load_weights(model_name)
    test_imgs = mpimg.imread(test_dir+img_list[0]).reshape((1,512,512,3))
    for i,img_name in enumerate(img_list):
        if i == 0: continue
        test_imgs = np.vstack((test_imgs,mpimg.imread(test_dir+img_name).reshape((1,512,512,3))))
    pred_imgs = model.predict(test_imgs)
    for i,img_name in enumerate(img_list):
        plt.imsave(predict_dir+img_name[0:4]+"_mask.png",reconstruct(pred_imgs[i]))

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
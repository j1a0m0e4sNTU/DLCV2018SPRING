import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,SGD
from keras import losses
from keras.utils import plot_model
import keras.backend as K 
import tensorflow as tf
from keras.metrics import binary_crossentropy
from keras.utils import plot_model

model = Sequential()
sgd = SGD(lr=1e-3,decay=1e-5)
optimizer = Adam(lr=1e-4)
epoch_num = 15
model_name = "model_last.h5"

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
    #load weights
    model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5",by_name=True)
    #FCN32
    model.add(Conv2D(filters=4096,kernel_size=(2,2),padding='same',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=4096,kernel_size=(1,1),padding='same',activation='relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=7,kernel_size=(1,1),padding='valid',activation='linear',kernel_initializer='he_normal'))
    model.add(Conv2DTranspose(filters=7,kernel_size=64,strides=32,padding='same',activation='softmax'))
    
    
    print("* Model is Builded ")

def load_train():
    train_x1 = np.load("train_x1.npy");print("train_x1.npy loaded") #(1200,512,512,3)
    train_y1 = np.load("train_y1.npy");print("train_y1.npy loaded") #(1200,512,512,8)
    train_x2 = np.load("train_x2.npy");print("train_x2.npy loaded") #(1114,512,512,3)
    train_y2 = np.load("train_y2.npy");print("train_y2.npy loaded") #(1114,512,512,8)
    train_x = np.vstack((train_x1,train_x2))
    train_y = np.vstack((train_y1,train_y2))
    train_y = np.stack((train_y[:,:,:,0],train_y[:,:,:,1],train_y[:,:,:,2],train_y[:,:,:,4],train_y[:,:,:,5],train_y[:,:,:,6],train_y[:,:,:,7]),axis=3) #(,512,512,7)
    return train_x,train_y

def load_valid():
    valid_x = np.load("valid_x.npy")
    valid_y = np.load("valid_y.npy")
    valid_y = np.stack((valid_y[:,:,:,0],valid_y[:,:,:,1],valid_y[:,:,:,2],valid_y[:,:,:,4],valid_y[:,:,:,5],valid_y[:,:,:,6],valid_y[:,:,:,7]),axis=3)
    return valid_x, valid_y

def train():
    train_x ,train_y = load_train()
    model.fit(train_x,train_y,batch_size=10,epochs=epoch_num)
    

def validate():
    valid_x, valid_y = load_valid()
    loss, accuracy = model.evaluate(valid_x,valid_y,batch_size=10)
    print(" VALIDATION --> loss =",loss,"acc =",accuracy)


def test():
    print("///////test///////")
    build_model()
    train_x = np.load("train_x1.npy");print("train_x1.npy loaded") #(1200,512,512,3)
    train_y = np.load("train_y1.npy");print("train_y1.npy loaded") #(1200,512,512,8)
    train_y = np.stack((train_y[:,:,:,0],train_y[:,:,:,1],train_y[:,:,:,2],train_y[:,:,:,4],train_y[:,:,:,5],train_y[:,:,:,6],train_y[:,:,:,7]),axis=3) #(,512,512,7)
    img = train_x[5].reshape((1,512,512,3))
    ans = train_y[5]
    ans = np.sum(ans,axis=(0,1))
    print("correct result: \n", ans)
    model.compile( optimizer= sgd, loss = 'categorical_crossentropy', metrics= ['accuracy'] )
    predict = model.predict(img).reshape((512,512,7))
    result = np.zeros((512,512,7),dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            result[i][j][np.argmax(predict[i][j])] = 1
    print("Prediction: \n",np.sum(result,axis=(0,1)))
    for i in range(10):
        model.fit(train_x[50*i:50*(i+1)],train_y[50*i:50*(i+1)],batch_size=10,epochs=1)
        predict = model.predict(img).reshape((512,512,7))
        result = np.zeros((512,512,7),dtype=np.uint8)
        for i in range(512):
            for j in range(512):
                result[i][j][np.argmax(predict[i][j])] = 1
        print("Prediction: \n",np.sum(result,axis=(0,1)))

def test2():
    build_model()
    x = np.zeros((1,512,512,3))
    y = model.predict(x)
    print(y.shape)

def main():
    print('=======main========') 
    build_model()
    model.compile( optimizer= optimizer, loss = 'categorical_crossentropy', metrics= ['accuracy'] )
    train()
    validate()
    model.save_weights(model_name)

def plot():
    build_model()
    #plot_model(model,'VGG16FCN.png',show_shapes=True,show_layer_names=True)
    model.summary()
if __name__ == '__main__':
    main()

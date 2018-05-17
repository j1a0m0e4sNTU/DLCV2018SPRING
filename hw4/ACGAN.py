import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose,MaxPooling2D,Embedding,multiply
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam,SGD
from keras import losses
from random import random
import csv

#####     parameters   ######

epoch_num = 1
input_dim = 1024
batch_size = 50
sgd = SGD(lr = 1e-4)
adam = Adam(lr= 1e-4)
noise_mean = 0.5
noise_var = 1

gd_rate = 2

D_size = 500
G_size = int(D_size*gd_rate)

losses = ['binary_crossentropy', 'binary_crossentropy']
################################

weights_name_load = 'acgan0515.h5'
weights_name_save = 'acgan0515.h5'

#####   Model & Layers ######

I = Input(shape=(input_dim, ) ) #last is for attribute
r  = Reshape(target_shape=(1,1,input_dim))
g0 = Conv2DTranspose(filters = 1024, kernel_size= 4, strides= 4,padding='same',activation='relu')
g1 = Conv2DTranspose(filters = 512, kernel_size= 4, strides= 4,padding='same',activation='relu')
g2 = Conv2D(filters=256,kernel_size=(2,2),padding='same',activation='relu')
g3 = Conv2DTranspose(filters = 128, kernel_size= 2, strides= 2,padding='same',activation='relu')
g4 = Conv2DTranspose(filters = 3, kernel_size= 2, strides= 2,padding='same',activation='sigmoid')
gen_layers = g4(g3(g2(g1(g0(r(I))))))


I2  = Input(shape=(64,64,3),dtype='float32')
d0 = Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu') 
d1 = Conv2D(filters=256,kernel_size=(2,2),padding='same',activation='relu') 
p1 = MaxPooling2D(pool_size= 4,strides= 4,padding='same')
d2 = Conv2D(filters=512,kernel_size=(2,2),padding='same',activation='relu') 
d3 = Conv2D(filters=512,kernel_size=(2,2),padding='same',activation='relu') 
p2 = MaxPooling2D(pool_size= 4,strides= 4,padding='same')
flat = Flatten()     #64*64*512
z = Dense(input_dim,activation='sigmoid')
validity = Dense(1,activation='sigmoid',name='vadility')
lebel    = Dense(1,activation='sigmoid',name='lebel')


ACGAN_generator = Model(I,gen_layers)
discriminator = Model(I2,[validity(z(flat(p2(d3(d2(p1(d1(d0(I2))))))))), lebel(z(flat(p2(d3(d2(p1(d1(d0(I2)))))))))])
acgan = Model(I,[validity(z(flat(p2(d3(d2(p1(d1(d0(gen_layers))))))))), lebel(z(flat(p2(d3(d2(p1(d1(d0(gen_layers)))))))))])


#################################


def pre(action = 0):
    if action == 0:
        ACGAN_generator.load_weights('VAE_decoder.h5')
        encoder = Model(I2,z(flat(p2(d3(d2(p1(d1(d0(I2)))))))))
        encoder.load_weights('VAE_encoder.h5')
    else:
        acgan.load_weights(weights_name_load)

def GetAtribute():
    data_train = []
    with open('hw4_data/train.csv',newline='') as csvfile:
        file = csv.reader(csvfile)
        for row in file:
            data_train.append(row[10])
    attribute = np.array(data_train[1:])
    
    attri_train = np.zeros((40000,1))
    attri_train[attribute == '1.0'] = [1]

    return attri_train
    
def train():
    print('=======  training...  =======')
    attri_train = GetAtribute() # (40000,1) 
    train_imgs = np.load('train.npy')
    
    for epoch in range(epoch_num):
        
        for step in range(40000//D_size):

            # train discriminator 
            for layer in discriminator.layers:
                layer.trainable = True
            discriminator.compile(optimizer=sgd,loss=losses,metrics=['accuracy'])
            
            print('Epoch:',epoch,'/',epoch_num,' Step:',step,'/',40000//D_size,'Training Discriminator with valid...')
            imgs = train_imgs[D_size*step:D_size*(step+1)]
            attri = attri_train[D_size*step:D_size*(step+1)]
            valid = np.ones((D_size,1))
            discriminator.fit(x = imgs, y=[attri,valid], batch_size=batch_size)

            print('Epoch:',epoch,'/',epoch_num,' Step:',step,'/',40000//D_size,'Training Discriminator with fake ...')
            noise = np.random.normal(noise_mean,noise_var,size=(D_size,input_dim))
            img_fake = ACGAN_generator.predict(noise)
            attri = np.ones((D_size,1))*0.5
            fake = np.zeros((D_size,1))
            discriminator.fit(x= img_fake, y= [fake,attri])

            # train generator
            for layer in discriminator.layers:
                layer.trainable = False
            acgan.compile(optimizer=adam,loss=losses,metrics=['accuracy'])

            print('Epoch:',epoch,'/',epoch_num,' Step:',step,'/',40000//D_size,'Training Generator...')
            noise = np.random.normal(noise_mean, noise_var, (G_size,input_dim))
            attri = np.random.randint(0,2,(G_size,1))
            noise[:,input_dim-1] = attri[:,0]
            valid = np.ones((G_size,1))
            acgan.fit(x= noise, y=[valid,attri], batch_size= batch_size)

            generate('ACGAN/',str(epoch)+'_'+str(step),2)

            if step % 20 == 0:
                 acgan.save_weights(weights_name_save)


def generate(dir_name,name,number):
    noise = np.random.normal(noise_mean,noise_var,(number,input_dim))
    noise[:,input_dim-1] = 1
    faces_pos = ACGAN_generator.predict(noise)
    noise[:,input_dim-1] = 0
    faces_neg = ACGAN_generator.predict(noise)
    for i in range(number):
        plt.imsave(dir_name+name+'_'+str(i)+'_pos.jpg',faces_pos[i])
        plt.imsave(dir_name+name+'_'+str(i)+'_neg.jpg',faces_neg[i])

if __name__ == '__main__':
    #pre(0)
    #train()
    acgan.load_weights('ACGAN.h5')
    ACGAN_generator.save_weights('ACGAN_generator.h5')
    
    
    
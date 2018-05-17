import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose,MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam,SGD
from keras import losses
from random import random

#####     parameters   ######
steps = 25

input_dim = 1024
batch_size = 20
epochs = 1
optimizer = SGD(lr = 1e-4)
noise_mean = 0.5
noise_var = 1

F_rate = 1
gd_rate = 10

valid_size = 500
fake_size = int(valid_size*F_rate)
gen_size = int(valid_size*gd_rate)
################################

weights_name_load = 'GAN.h5'
weights_name_save = 'gan0511.h5'

#####   Model & Layers ######
I = Input(shape=(input_dim, ) )
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
o = Dense(1,activation='sigmoid')
dis_layers = o(z(flat(p2(d3(d2(p1(d1(d0(I2)))))))))
gan_layers = o(z(flat(p2(d3(d2(p1(d1(d0(gen_layers)))))))))

GAN_generator = Model(I,gen_layers)
discriminator = Model(I2,dis_layers)
gan = Model(I,gan_layers)

#################################


def pre():
    GAN_generator.load_weights('VAE_decoder.h5')
    enco = Model(I2,z(flat(p2(d3(d2(p1(d1(d0(I2)))))))))
    enco.load_weights('VAE_encoder.h5')

def comp():
    discriminator.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    gan.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

def train():
    print('=======  training...  =======')
    train_imgs = np.load('train.npy')
    valid_D = np.ones((valid_size,1))
    fake_D = np.zeros((fake_size,1))
    valid_G = np.ones((gen_size,1))

    for step in range(steps):
        # train discriminator
        for layer in discriminator.layers:
            layer.trainable = True
        comp()
        print('Step:',step,'Training Discriminator with valid...')
        img_ids = np.random.randint(0,40000,valid_size)
        discriminator.fit(x = train_imgs[img_ids], y = valid_D, batch_size= batch_size)
        
        print('Step:',step,'Training Discriminator with fake...')
        noise = np.random.normal(noise_mean,noise_var, size=(fake_size,input_dim) )
        gen_imgs = GAN_generator.predict(noise)
        discriminator.fit(x = gen_imgs, y = fake_D )

        # train generator
        for layer in discriminator.layers:
            layer.trainable = False
        comp()

        print('Step:',step,'Training Generator...')
        noise = np.random.normal(noise_mean,noise_var,size=(gen_size,input_dim))
        gan.fit(x = noise, y = valid_G,batch_size= batch_size)

        generate('GAN/',str(step),2)

    gan.save_weights(weights_name_save)
    
    
    

def generate(dir_name,name,number):
    noise = np.random.normal(noise_mean,noise_var,(number,input_dim))
    faces = GAN_generator.predict(noise)
    for i in range(number):
        plt.imsave(dir_name+name+'_'+str(i)+'.jpg',faces[i])


if __name__ == '__main__':
    gan.load_weights(weights_name_load)
    #pre()
    train()
 

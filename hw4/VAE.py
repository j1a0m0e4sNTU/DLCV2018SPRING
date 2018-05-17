import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose,MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam,SGD
from keras import losses
from random import random


#### parameters #####
latent_dim = 1024
batch_size = 10
epochs = 8
optimizer = Adam(lr = 1e-4)
KL_rate = 1e-5
weights_name_load = 'vae0511.h5'
weights_name_save = 'vae0511.h5'

#####################

####  Model laysers  #####

I  = Input(shape=(64,64,3),dtype='float32')
e0 = Conv2D(filters=128,kernel_size=(2,2),padding='same',activation='relu') 
e1 = Conv2D(filters=256,kernel_size=(2,2),padding='same',activation='relu') 
p1 = MaxPooling2D(pool_size= 4,strides= 4,padding='same')
e2 = Conv2D(filters=512,kernel_size=(2,2),padding='same',activation='relu') 
e3 = Conv2D(filters=512,kernel_size=(2,2),padding='same',activation='relu') 
p2 = MaxPooling2D(pool_size= 4,strides= 4,padding='same')
flat = Flatten()(p2((e3(e2(p1(e1(e0(I))))))))   #64*64*512
z_mean = Dense(latent_dim,activation='sigmoid')(flat)
z_var  = Dense(latent_dim)(flat)
epsilon = K.random_normal(shape = (latent_dim,), mean = 0, stddev=1.0)
def sampling(args):
    mean,var = args
    return mean+K.exp(var)*epsilon
z = Lambda(function = sampling,output_shape=(latent_dim,))([z_mean, z_var]) #latent space

I2 = Input(shape=(latent_dim, ) )
r  = Reshape(target_shape=(1,1,latent_dim))
d0 = Conv2DTranspose(filters = 1024, kernel_size= 4, strides= 4,padding='same',activation='relu')
d1 = Conv2DTranspose(filters = 512, kernel_size= 4, strides= 4,padding='same',activation='relu')
d2 = Conv2D(filters=256,kernel_size=(2,2),padding='same',activation='relu')
d3 = Conv2DTranspose(filters = 128, kernel_size= 2, strides= 2,padding='same',activation='relu')
d4 = Conv2DTranspose(filters = 3, kernel_size= 2, strides= 2,padding='same',activation='sigmoid')
VAE = d4(d3(d2(d1(d0(r(z))))))

vae = Model(I,VAE)
encoder = Model(I,z_mean)
decoder = Model(I2,d4(d3(d2(d1(d0(r(I2)))))))

######  Loss  #######
kl_loss = - 0.5 * K.sum(1 + z_var - K.square(z_mean) - K.exp(z_var), axis=-1)
vae_loss = KL_rate * kl_loss

###### Metrics ######
def mse(y_true,y_pred):
    return K.mean(K.square(y_true-y_pred))
#####################
def train():
    print("--------  training...   --------")
    vae = Model(I,VAE)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizer,loss='mse',metrics= ['accuracy',mse ] )
    train = np.load('train.npy') 
    vae.fit(x= train,y=train, batch_size=batch_size, epochs= epochs )
    vae.save_weights(weights_name_save)
    
    test = np.load('test.npy')
    loss, acc = vae.evaluate(test,test, batch_size= batch_size)
    print(" Validation Result :\nloss :",loss,'\nacc :',acc)  

def reconstruct( ):
    print("-------  reconstructing...  ------")
    global epsilon
    epsilon = 0
    test_dir = 'hw4_data/reconstruct/'
    vae.load_weights(weights_name_load)
    test = np.load('test.npy')
    test_num,_,_,_ = test.shape
    recons = vae.predict(test)
    for i in range(test_num):
        plt.imsave(test_dir +str(i)+'.jpg',recons[i])
    mse = np.mean(np.square(test-recons))
    print('the mse of test set =',mse)

def generate():
    print("-------  generating...  --------") 
    gen_num = 200
    gen_dir = 'hw4_data/generate/'
    vae.load_weights(weights_name_load)
    rands = np.random.normal(0.5,2,(gen_num,latent_dim))
    gen = decoder.predict(rands)
    for i in range(gen_num):
        plt.imsave(gen_dir +str(i)+'.jpg', gen[i])

def split_wight():
    vae.load_weights(weights_name_load)
    encoder.save_weights('VAE_encoder.h5')
    decoder.save_weights('VAE_decoder.h5')
    vae.save_weights('VAE.h5')

if __name__ == '__main__':
    split_wight()

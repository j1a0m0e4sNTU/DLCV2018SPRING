from VAE import vae,decoder,encoder
from GAN import GAN_generator
from ACGAN import ACGAN_generator
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import csv
import os
import sys

np.random.seed(1004)
###########       VAE       #############
def fig1_2(dest_dir):
    epoch = [0.5,1,2,3,4,5,6,7,8]
    mse = [0.03,0.0235,0.0152,0.0133,0.0121,0.0114,0.0108,0.0103,0.0099]
    kl  = [0.0,0.0022,0.0026,0.0027,0.0028,0.0028,0.0027,0.0027,0.0027]
    plt.subplot(1,2,1)
    plt.title('KLD')
    plt.plot(epoch,kl,'r')
    plt.xlabel('epochs')

    plt.subplot(1,2,2)
    plt.title('MSE')
    plt.plot(epoch, mse,'r')
    plt.xlabel('epochs')

    plt.savefig(dest_dir+'fig1_2.jpg'); plt.close()

def fig1_3(source_dir,dest_dir):
    test_names = [img for img in os.listdir(source_dir+'test/')]; test_names.sort()
    img_test = mpimg.imread(source_dir+'test/'+test_names[0]).reshape(1,64,64,3)
    for i in range(1,10):
        img_test = np.vstack((img_test,mpimg.imread(source_dir+'test/'+test_names[i]).reshape(1,64,64,3)))

    img_recons = vae.predict(img_test)

    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(img_test[i])
        plt.axis('off')
        plt.subplot(2,10,i+11)
        plt.imshow(img_recons[i])
        plt.axis('off')

    plt.savefig(dest_dir+'fig1_3.jpg'); plt.close()

def fig1_4(dest_dir):
    noise = np.random.normal(0.5,1,(32,1024))
    img_gen = decoder.predict(noise)

    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(img_gen[i])
        plt.axis('off')

    plt.savefig(dest_dir+'fig1_4.jpg'); plt.close()

def fig1_5(source_dir,dest_dir): 
    test_names = [img for img in os.listdir(source_dir+'test/')]; test_names.sort()
    img_test = mpimg.imread(source_dir+'test/'+test_names[0]).reshape(1,64,64,3)
    for i in range(1,200):
        img_test = np.vstack((img_test,mpimg.imread(source_dir+'test/'+test_names[i]).reshape(1,64,64,3)))

    data_test = []
    with open(source_dir+'test.csv',newline='') as csvfile:
        file = csv.reader(csvfile)
        for row in file:
            data_test.append(row[8])
    attribute = np.array(data_test[1:201])

    pos   = img_test[attribute == '1.0' ]
    neg = img_test[attribute == '0.0' ]
    pos_num = pos.shape[0]

    people = np.vstack((pos,neg))
    latents = encoder.predict(people)
    
    dots = TSNE(n_components = 2).fit_transform(latents)
    plt.scatter(dots[:pos_num,0],dots[:pos_num,1],c = 'b')
    plt.scatter(dots[pos_num:,0],dots[pos_num:,1],c = 'r')
    plt.savefig(dest_dir+'fig1_5.jpg'); plt.close()

##################  GAN  ######################
def fig2_2(dest_dir):
    epoch = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    valid = [0.6238,0.6274,0.6367,0.6408,0.6488,0.6558,
             0.6609,0.6722,0.6669,0.6846,0.6845,0.6734,
             0.6756,0.6951,0.6928,0.6811,0.6886,0.6919,
             0.7048,0.6954,0.7003,0.6844,0.7039,0.7085,0.7055]
    fake  = [0.7909,0.7951,0.7912,0.7829,0.7824,0.7839,
             0.7643,0.7546,0.7604,0.7522,0.7480,0.7483,
             0.7607,0.7548,0.7576,0.7639,0.7555,0.7714,
             0.7563,0.7665,0.7582,0.7625,0.7591,0.7502,0.7454]
    gen   = [0.6428,0.6553,0.6627,0.6695,0.6704,0.6781,
             0.6783,0.6793,0.6811,0.6839,0.6837,0.6786,
             0.6825,0.6811,0.6835,0.6869,0.6882,0.6851,
             0.6797,0.6851,0.6847,0.6891,0.6869,0.6864,0.6869]
    plt.plot(epoch,valid,'r')
    plt.plot(epoch,fake,'g')
    plt.plot(epoch,gen,'b')
    plt.title('Learning Curve of GAN')
    plt.savefig(dest_dir+'fig2_2.jpg'); plt.close()

def fig2_3(dest_dir):
    noise = np.random.normal(0.5,1,(32,1024))
    img_gen = GAN_generator.predict(noise)

    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(img_gen[i])
        plt.axis('off')

    plt.savefig(dest_dir+'fig2_3.jpg'); plt.close()

###########       ACGAN      ##############
def fig3_2(dest_dir):
    epoch = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    v_v = [0.6692,0.6879,0.6890,0.6733,0.6765,0.6837,0.6812,0.6933,0.6843,0.6976,
           0.7057,0.7158,0.7118,0.7105,0.7045,0.7074,0.7168,0.7272,0.7176,0.7241]
    v_l = [0.6718,0.6798,0.6654,0.6863,0.6743,0.6726,0.6877,0.6663,0.6764,0.6766,
           0.6636,0.6633,0.6535,0.6584,0.6667,0.6701,0.6684,0.6705,0.6739,0.6642]
    f_v = [0.7059,0.7170,0.7017,0.6974,0.6972,0.6828,0.7003,0.6921,0.6724,0.6908,
           0.6895,0.6845,0.6959,0.6861,0.7023,0.6883,0.6843,0.6889,0.6800,0.6740]
    f_l = [0.7045,0.7057,0.6950,0.7058,0.7144,0.7119,0.7128,0.6976,0.7062,0.6972,
           0.7125,0.7051,0.7078,0.7063,0.6909,0.7152,0.7088,0.7022,0.7211,0.7148]
    g_v = [0.7439,0.7471,0.7554,0.7535,0.7526,0.7454,0.7595,0.7565,0.7625,0.7592,
           0.7585,0.7664,0.7644,0.7566,0.7593,0.7571,0.7649,0.7591,0.7704,0.7613]
    g_l = [0.7122,0.7112,0.6995,0.7083,0.7092,0.6960,0.7055,0.7049,0.7062,0.7047,
           0.7072,0.7044,0.7076,0.7054,0.7093,0.7061,0.7045,0.7074,0.7110,0.7032]

    plt.subplot(3,1,1); plt.title('Discriminator with True Image')
    plt.plot(epoch,v_v,'r')
    plt.plot(epoch,v_l,'b')
    plt.xlabel('epochs')

    plt.subplot(3,1,2); plt.title('Discriminator with Fake Image')
    plt.plot(epoch,f_v,'r')
    plt.plot(epoch,f_l,'b')
    plt.xlabel('epochs')

    plt.subplot(3,1,3); plt.title('Generator Loss')
    plt.plot(epoch,v_v,'r')
    plt.plot(epoch,v_l,'b')
    plt.xlabel('epochs')

    plt.savefig(dest_dir+'fig3_2.jpg');plt.close()

def fig3_3(dest_dir):
    noise = np.random.normal(0.5,1,(10,1024))
    noise[:,1023] = 1
    img_pos = ACGAN_generator.predict(noise)
    noise[:,1023] = 0
    img_neg = ACGAN_generator.predict(noise)

    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(img_pos[i])
        plt.axis('off')
        plt.subplot(2,10,i+11)
        plt.imshow(img_neg[i])
        plt.axis('off')

    plt.savefig(dest_dir+'fig3_3.jpg'); plt.close()


if __name__ == '__main__':
    vae.load_weights('VAE.h5')
    GAN_generator.load_weights('GAN_generator.h5')
    ACGAN_generator.load_weights('ACGAN_generator.h5')

    source_dir = sys.argv[1] +'/'
    dest_dir   = sys.argv[2][:-1] +'/'
    
    fig1_2(dest_dir)
    fig1_3(source_dir,dest_dir)
    fig1_4(dest_dir)
    fig1_5(source_dir,dest_dir)
    fig2_2(dest_dir)
    fig2_3(dest_dir)
    fig3_2(dest_dir)
    fig3_3(dest_dir)
    

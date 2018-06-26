import numpy as np
import sys
import os
import torch
from torchvision import transforms, models
from reader import readShortVideo, getVideoList
import skimage.io as io
import skimage.transform as skt
from RNN_seq import RNN_predictor

## parameters ##
vdo_dir = sys.argv[1]
txt_dest = sys.argv[2]

batch_size = 20
model_name = 'rnn_seq.pkl'

def get_array(dir_name):
    ''' 
    Return numpy array of all images of given directory name, shape = (img_num,120,160,3)
    ''' 
    img_names = [ dir_name+img for img in os.listdir(dir_name) ]
    img_names.sort()
    img_num = len(img_names)
    img_arr = np.zeros((img_num,120,160,3),dtype= np.uint8)
    for img_id in range(img_num):
        img = io.imread(img_names[img_id])
        img = skt.rescale(img,0.5,mode='constant',preserve_range=True).astype(np.uint8)
        img_arr[img_id] = img

    return img_arr

def get_features(dir_name): #input: each vdo directory
    '''Preprocess each vdo, transform into the shape of ( img_num, feature_dim = 512*3*5)'''

    extractor = models.vgg16(pretrained= True).features.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    imgs = get_array(dir_name)
    imgs = np.transpose(imgs,(0,3,1,2))  #(img_num,3,120,160)
    imgs = torch.tensor(imgs,dtype = torch.float)
    imgs = imgs/255
    for i in range(imgs.shape[0]):
        imgs[i] = normalize(imgs[i])
    # now imgs are tensors in shape of (img_num,3,120,160), and normalized
    img_num = imgs.shape[0]
    batch_id = 0  
    vdo_features = torch.zeros((img_num , 512*3*5),dtype = torch.float)
    while batch_id*batch_size < img_num:
        start = batch_id*batch_size
        end   = (batch_id+1)*batch_size if  (batch_id+1)*batch_size < img_num else img_num
            
        img_batch = imgs[ start: end].cuda() #(batch_size,3,120,160)
        feature_batch = extractor(img_batch) # (batch_size,512,3,5)
        feature_batch = feature_batch.view(end-start,-1).detach().cpu()#(batch_size,512*3*5)
        vdo_features[start : end ] = feature_batch
        batch_id += 1 
    print('Finished extracting features of',dir_name)
    return vdo_features

def main():
    vdo_dirs = [ name for name in os.listdir(vdo_dir) ]; vdo_dirs.sort()

    for name in vdo_dirs:

        vdo = get_features(vdo_dir+'/'+name+'/')
        img_num = vdo.shape[0]
        # now get features (vdo_num, 2048,7,10)
    
        txt = open(txt_dest+'/'+name+'.txt','w')
        predictor = RNN_predictor().cuda()
        predictor.load_state_dict(torch.load(model_name))
        result = torch.zeros((img_num ,))
        h_n = torch.zeros((1,1,1024),dtype=torch.float).cuda()
        h_c = torch.zeros((1,1,1024),dtype=torch.float).cuda()

        result = torch.zeros((img_num,1),dtype = torch.long)
        for img_id in range(img_num):
            in_x = vdo[img_id].view(1,1,-1).cuda()
            h_n.detach_(); h_c.detach_()
            out_x, (h_n , h_c) = predictor(in_x, (h_n,h_c))
            result[img_id] = torch.max(out_x,1)[1]
        # now get pridicting result (vdo_num,)
        for i in range(img_num):
            txt.write(str(int(result[i]))+'\n')

        txt.close()
        print('Finished predicting',name)


if __name__ == '__main__':
    print('==========  HW5 p3 =============')
    main()
    

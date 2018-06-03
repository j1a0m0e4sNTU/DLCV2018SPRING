import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import sys
import os
import skimage.io as io
import skimage.transform as skt
import matplotlib.pyplot as plt

####  data path  ####
train_videos_dir = 'data/FullLengthVideos/videos/train/'
train_labels_dir = 'data/FullLengthVideos/labels/train/'
train_vdo_names = [train_videos_dir+name+'/' for name in os.listdir(train_videos_dir) ];train_vdo_names.sort()
train_label_files = [train_labels_dir+name for name in os.listdir(train_labels_dir) ];train_label_files.sort()

valid_videos_dir = 'data/FullLengthVideos/videos/valid/'
valid_labels_dir = 'data/FullLengthVideos/labels/valid/'
valid_vdo_names = [valid_videos_dir+name+'/' for name in os.listdir(valid_videos_dir) ];valid_vdo_names.sort()
valid_label_files = [valid_labels_dir+name for name in os.listdir(valid_labels_dir) ];valid_label_files.sort()
####  parameters ####
epoch_num = 10
batch_size = 20
lr = 1e-4
save_name = 'rnn_seq0601.pkl'
load_name = 'rnn_seq0601.pkl' #'rnn0531_2.pkl'
####    model    ####
class RNN_predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = 512*3*5,
            hidden_size=1024,
            num_layers = 1,
            batch_first = True
        )

        self.fc1  = nn.Linear(in_features = 1024,out_features=16)
        self.relu1 = nn.ReLU(inplace= True)
        self.fc2 = nn.Linear(in_features = 16, out_features = 11)
        self.soft= nn.Softmax(dim=1)

    def forward(self, x, h):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size = hidden size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, h)
        out = self.fc1(r_out[:,-1,:]) #take the last output
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out, (h_n,h_c)

####  functions  ####
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

def Extract_features(mode = 'train'):
    '''Preprocess each vdo, transform into the shape of ( img_num, feature_dim = 512*3*5)'''

    extractor = models.vgg16(pretrained= True).features.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    vdo_names = train_vdo_names if mode == 'train' else valid_vdo_names
    for vdo_id in range(len(vdo_names)):
        imgs = get_array(vdo_names[vdo_id])
        imgs = np.transpose(imgs,(0,3,1,2))
        imgs = torch.tensor(imgs,dtype = torch.float)
        imgs = imgs/255
        for i in range(imgs.shape[0]):
            imgs[i] = normalize(imgs[i])
        # now imgs are tensors in shape of (img_num,3,120,160), and normalized
        img_num = imgs.shape[0]
        batch_id = 0  
        vdo_features = np.zeros((img_num , 512*3*5),dtype = np.float)
        while batch_id*batch_size < img_num:
            start = batch_id*batch_size
            end   = (batch_id+1)*batch_size if  (batch_id+1)*batch_size < img_num else img_num
            
            img_batch = imgs[ start: end].cuda() #(batch_size,3,120,160)
            feature_batch = extractor(img_batch) # (batch_size,512,3,5)
            feature_batch = feature_batch.view(end-start,-1).cpu().detach().numpy() #numpy (10,512*3*5)
            
            vdo_features[start : end ] = feature_batch
            batch_id += 1 
        
        np.save('task3/vdo_'+mode+str(vdo_id)+'.npy',vdo_features)
        print('task3/vdo_'+mode+str(vdo_id)+'.npy is saved!')

def Extract_labels(mode = 'train'):
    label_files = train_label_files if mode == 'train' else valid_label_files
    for file_id in range(len(label_files)): 
        file = open(label_files[file_id])
        file = list(file)
        file = np.array(file)
        
        label = np.zeros((file.shape[0],))
        for i in range(11):
            label[file == str(i)+'\n'] = i
        np.save('task3/labels_'+mode+str(file_id)+'.npy',label)
        print('task3/labels_'+mode+str(file_id)+'.npy is saved!')
        
def load_files(mode = 'train'):
    vdos = ['task3/'+file for file in os.listdir('task3/') if file.startswith('vdo_'+mode)]
    labels = ['task3/'+file for file in os.listdir('task3/') if file.startswith('labels_'+mode)]
    vdos.sort(); labels.sort()
    return vdos, labels, len(vdos)

def train(retrain = False):
    ## settings
    vdos, labels, vdo_num = load_files('train')
    predictor = RNN_predictor().cuda()
    if(retrain):
        print('Retraining...')
        predictor.load_state_dict(torch.load(load_name))
    else: print('Training...')
    
    optimizer = optim.Adam(params=predictor.parameters(),lr=lr)
    CrossEntropy = nn.CrossEntropyLoss()
    ## start training
    for epoch in range(epoch_num): #
        print()
        for vdo_id in range(vdo_num): #
            print('Epoch',epoch,vdos[vdo_id],'----> ',end='')
            vdo = np.load(vdos[vdo_id])
            vdo = torch.tensor(vdo,dtype= torch.float)
            label = np.load(labels[vdo_id])
            label = torch.tensor(label,dtype = torch.long).view(-1,1)
            h_n = torch.zeros((1,1,1024),dtype=torch.float).cuda()
            h_c = torch.zeros((1,1,1024),dtype=torch.float).cuda()
            LOSS = 0
            img_num = vdo.shape[0]
            for img_id in range(img_num):
                in_x = vdo[img_id].view(1,1,-1).cuda()
                h_n.detach_(); h_c.detach_()
                out_x, (h_n , h_c) = predictor(in_x, (h_n,h_c))
                optimizer.zero_grad()

                loss = CrossEntropy(out_x, label[img_id].cuda())
                loss.backward()
                optimizer.step()
                LOSS += loss.item()

            print('Average loss =',LOSS/img_num)

    torch.save(predictor.state_dict(),save_name)

def evaluate(mode = 'train'):
    print('Testing Performance on '+mode +' set...')
    vdos, labels, vdo_num = load_files(mode)
    predictor = RNN_predictor().cuda()
    predictor.load_state_dict(torch.load(load_name))
    
    for vdo_id in range(vdo_num):
        vdo = np.load(vdos[vdo_id])
        vdo = torch.tensor(vdo,dtype= torch.float)
        label = np.load(labels[vdo_id])
        label = torch.tensor(label,dtype = torch.long).view(-1,1)
        h_n = torch.zeros((1,1,1024),dtype=torch.float).cuda()
        h_c = torch.zeros((1,1,1024),dtype=torch.float).cuda()
        
        img_num = vdo.shape[0]
        result = torch.zeros((img_num,1),dtype = torch.long)
        for img_id in range(img_num):
            in_x = vdo[img_id].view(1,1,-1).cuda()
            h_n.detach_(); h_c.detach_()
            out_x, (h_n , h_c) = predictor(in_x, (h_n,h_c))
            result[img_id] = torch.max(out_x,1)[1]
        #get prediction, then evaluate
        correct = label[label == result].shape[0]
        print('Accuracy of ',vdos[vdo_id],'=',correct/img_num)


def test():
    print('testing...')
    

if __name__ == '__main__':
    print('========== RNN from sequuence to sequence ===========')

    if  sys.argv[1] == 'extract':
        Extract_features('train')
        Extract_features('valid')
        Extract_labels('train')
        Extract_labels('valid')

    elif sys.argv[1] == 'train':
        train(False)
    
    elif sys.argv[1] == 'retrain':
        train(True)
    
    elif sys.argv[1] == 'evaluate':
        evaluate('train')
        #evaluate('valid')
    
    elif sys.argv[1] == 'test':
        test()
    
    else :
        print('Command not found!!')
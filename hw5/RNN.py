from reader import readShortVideo, getVideoList
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, models
import sys
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
####  data path  ####
train_vdo_dir = 'data/TrimmedVideos/video/train'
valid_vdo_dir = 'data/TrimmedVideos/video/valid'
gt_train = 'data/TrimmedVideos/label/gt_train.csv'
gt_valid = 'data/TrimmedVideos/label/gt_valid.csv'

####  parameters ####
epoch_num = 20
batch_size = 10
lr = 1e-4
save_name = 'rnn0531_2.pkl'
load_name = 'rnn0531_2.pkl'

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

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size = hidden size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc1(r_out[:,-1,:]) #take the last output
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out

####  functions  ####

def Extract_features(mode='train'):
    extractor = models.vgg16(pretrained= True).features.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    csv = 0
    if mode == 'train':
        csv = getVideoList(gt_train)
    else: # validation
        csv = getVideoList(gt_valid)
    
    vdo_num = len(csv['Video_index'])
    frames = 0      ##vdo-frames-holder
    frames_num = 10 ## number of frames per clip 
    vdo_features = np.zeros((vdo_num,frames_num,512*3*5),dtype= np.float)

    for clip_id in range(vdo_num): 
        print(mode,clip_id)
        if mode == 'train':
            frames = readShortVideo(train_vdo_dir,csv['Video_category'][clip_id],csv['Video_name'][clip_id],2,0.5)
        else : #validation
            frames = readShortVideo(valid_vdo_dir,csv['Video_category'][clip_id],csv['Video_name'][clip_id],2,0.5)
        # frames is in shape of (num, 120,160,3)
        vdo_size = frames.shape[0] #get length of vdo, want to sample 40 frames of it
        skip = vdo_size/frames_num
        frame_sample = np.zeros((frames_num,120,160,3))
        for i in range(frames_num):
            frame_sample[i] = frames[int(i*skip)]
        frame_sample = np.transpose(frame_sample,(0,3,1,2))
        frame_sample = torch.tensor(frame_sample,dtype= torch.float).cuda()
        # sample completed --> (frames_num,3,120,160)per clip
        # now normalize it 
        frame_sample = (frame_sample/255)
        for i in range(frames_num):
            frame_sample[i] = normalize(frame_sample[i])
        
        feature_batchframe_features = extractor(frame_sample)  
            
        frame_features = frame_features.view(frames_num,-1)  ## in shape of (frames_num,512*3*5)
        
        vdo_features[clip_id] = frame_features.cpu().detach().numpy()
    
    np.save('rnn_'+mode+'_feature.npy',vdo_features) #shape = (vdo_num, seq = frames_num, 512*h*w)

def load_data(mode= 'train'):
    if mode == 'train':
        features = np.load('rnn_train_feature.npy')
        features = torch.tensor(features,dtype= torch.float)
        labels = np.load('train_ActionLabels.npy')
        labels = torch.tensor(labels,dtype= torch.long)
        labels = torch.max(labels,1)[1]
        print(mode,'data is loaded')
        return features,labels
    else:  #validation
        features = np.load('rnn_valid_feature.npy')
        features = torch.tensor(features,dtype= torch.float)
        labels = np.load('valid_ActionLabels.npy')
        labels = torch.tensor(labels,dtype= torch.long)
        labels = torch.max(labels,1)[1]
        print(mode,'data is loaded')
        return features,labels

def train(retrain = False):
    print('training...')
    ## load data 
    features, labels = load_data('train')
    ## settings 
    predictor = RNN_predictor().cuda()
    predictor.train()
    if retrain == True:
        predictor.load_state_dict(torch.load(load_name))
    predictor.train()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=predictor.parameters(), lr= lr)
    train_size = features.size(0)
    ## training 
    for epoch in range(epoch_num):
        batch_id = 0
        loss_rate = 0
        while batch_id*batch_size < train_size:
            start = batch_id*batch_size
            end   = train_size if (batch_id+1)*batch_size > train_size else  (batch_id+1)*batch_size
            in_batch = features[start : end].cuda()
            gt_batch = labels[start : end ].cuda()

            optimizer.zero_grad()

            out_batch = predictor(in_batch)
            loss = cross_entropy(out_batch,gt_batch)
            loss.backward()
            optimizer.step()

            loss_rate += loss.item()
            batch_id += 1

            if batch_id%50 == 0:
                print('Epoch',epoch,'after training',batch_id+1,'/',train_size//batch_size +1,' --> loss = ',loss_rate/(train_size//batch_size))
        print()   
    torch.save(predictor.state_dict(),save_name)

def evaluate(mode = 'training'):
    print('testing performance...')
    predictor = RNN_predictor().cuda()
    predictor.load_state_dict(torch.load(load_name))
    predictor.eval()
    features, labels = 0,0
    if mode == 'training':
        features, labels = load_data('train')
    else:
        features, labels = load_data('valid')

    size = labels.shape[0]
    label_result = torch.zeros(size)
    batch_id = 0
    while  batch_id*batch_size < size:
        start = batch_id*batch_size 
        end = size if (batch_id+1)*batch_size > size else (batch_id+1)*batch_size
        in_batch = features[start:end].cuda()
        out_batch= predictor(in_batch)

        label_batch = torch.max(out_batch,1)[1]
        label_result[start:end] = label_batch
        batch_id += 1
    
    correct_total = 0
    acc = torch.zeros(11,dtype=torch.float)
    for i in range(11):
        gt_num = labels[ labels == i].shape[0]
        result_each = label_result[ labels == i]
        correct = result_each[ result_each == i].shape[0]
        correct_total += correct
        acc[i] = correct/gt_num

    print("******  Performace on",mode,'set  *********')
    print("Accuracy for each category:\n", acc)
    print("Average Accuracy across each class:",torch.sum(acc)/11)
    print("Average Accuracy overall:",correct_total/size)
    

def test():
    print('testing....')
    model = models.vgg16(pretrained= True).features
    i = torch.zeros(1,3,120,160)
    f = model(i)
    print(f.shape)


if __name__ == '__main__':
    print('===========  RNN  ===========')
    if sys.argv[1] == 'train':
        train(False)
    
    elif sys.argv[1] == 'retrain':
        train(True)
    
    elif sys.argv[1] == 'evaluate':
        evaluate('training')
        evaluate('validation')
    
    elif sys.argv[1] == 'test':
        test()
        evaluate('validation')
        
    else :
        print('Command not found!!')
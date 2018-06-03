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
batch_size = 20

save_name = 'cnn_0528.pkl'
load_name = 'cnn_0528.pkl'

####  class  #####
class ActionPredictor(nn.Module):   #input  shape: (num,2048,7,10)
    def __init__(self):             #output shape: (num,11)
        super().__init__()
        self.fc1   = nn.Linear(in_features= 2048*7*10,out_features= 2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(p = 0.5)

        self.fc2   = nn.Linear(in_features=2048,out_features=1024)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p= 0.5)

        self.fc3   = nn.Linear(in_features=1024,out_features = 11)
        self.soft  = nn.Softmax(dim=1)
        
    def forward(self,x): 
        x = x.view(x.size(0),-1)  #flatten
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x) 
        x = self.soft(x)
        return x

#### functions ####
def Extract_features(mode='train'):
    extractor = models.vgg16(pretrained= True).features.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    csv = 0
    if mode == 'train':
        csv = getVideoList(gt_train)
    else: # validation
        csv = getVideoList(gt_valid)
    
    frames = 0
    vdo_features = 0
    for clip_id in range(len(csv['Video_index'])): #
        print(mode,clip_id)
        if mode == 'train':
            frames = readShortVideo(train_vdo_dir,csv['Video_category'][clip_id],csv['Video_name'][clip_id],5)
        else : #validation
            frames = readShortVideo(valid_vdo_dir,csv['Video_category'][clip_id],csv['Video_name'][clip_id],5)
        skip_num = frames.shape[0]//4     # 4 is frame number after sampling
        frame_sample = np.expand_dims(frames[0],axis=0)
        for i in range(1,4):
            frame_sample = np.vstack((frame_sample,np.expand_dims(frames[i*skip_num],axis=0)))
        ### now frame_sample is in shape of (4,240,320,3)
        frame_sample = np.transpose(frame_sample,(0,3,1,2))
        frame_sample = torch.tensor(frame_sample,dtype = torch.float).cuda()
        ### now frame_sample is transformed to tensor in shape of (4,3,240,320)
        ###normalize
        frame_sample = (frame_sample / 255)
        for i in range(4):
            frame_sample[i] = normalize(frame_sample[i])
        
        frame_features = extractor(frame_sample)        ### in shape of (4,512,7,10)
        frame_features = frame_features.view(1,-1,7,10) ### in shape of (1,2048,7,10) --> input for ActionPredictor for 1 clip
        
        if clip_id == 0:
            vdo_features = frame_features.cpu().detach().numpy()
        else:
            vdo_features = np.vstack((vdo_features,frame_features.cpu().detach().numpy()))

    np.save(mode+'_feature.npy',vdo_features)

     
def Extract_labels(mode = 'train'):
    csv = 0
    if mode == 'train':
        csv = getVideoList(gt_train)
    else: # validation
        csv = getVideoList(gt_valid)
    
    labels = np.array(csv['Action_labels'])
    actions = np.zeros((len(labels),11))

    actions[labels == '0'] = [1,0,0,0,0,0,0,0,0,0,0]
    actions[labels == '1'] = [0,1,0,0,0,0,0,0,0,0,0]
    actions[labels == '2'] = [0,0,1,0,0,0,0,0,0,0,0]
    actions[labels == '3'] = [0,0,0,1,0,0,0,0,0,0,0]
    actions[labels == '4'] = [0,0,0,0,1,0,0,0,0,0,0]
    actions[labels == '5'] = [0,0,0,0,0,1,0,0,0,0,0]
    actions[labels == '6'] = [0,0,0,0,0,0,1,0,0,0,0]
    actions[labels == '7'] = [0,0,0,0,0,0,0,1,0,0,0]
    actions[labels == '8'] = [0,0,0,0,0,0,0,0,1,0,0]
    actions[labels == '9'] = [0,0,0,0,0,0,0,0,0,1,0]
    actions[labels == '10'] = [0,0,0,0,0,0,0,0,0,0,1]

    np.save(mode+'_ActionLabels.npy',actions)

#################  train/validation  ###################
def load_data(mode= 'train'):
    if mode == 'train':
        features = np.load('train_feature.npy')
        features = torch.tensor(features,dtype= torch.float)
        labels = np.load('train_ActionLabels.npy')
        labels = torch.tensor(labels,dtype= torch.long)
        labels = torch.max(labels,1)[1]
        print(mode,'data is loaded')
        return features,labels
    else:  #validation
        features = np.load('valid_feature.npy')
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
    predictor = ActionPredictor().cuda()
    if retrain == True:
        predictor.load_state_dict(torch.load(load_name))
    predictor.train()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=predictor.parameters(), lr= 1e-3)
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

            if batch_id%20 == 0:
                print('Epoch',epoch,'after training',batch_id+1,'/',train_size//batch_size +1,' --> loss = ',loss_rate/(train_size//batch_size))
            
    torch.save(predictor.state_dict(),save_name)
            

def evaluate(mode = 'training'):
    print('testing performance...')
    predictor = ActionPredictor().cuda()
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
    print('testing...')
    

    

if __name__ == '__main__':
    print('============   CNN  =============')
    if sys.argv[1] == 'train':
        train(retrain=False)

    elif sys.argv[1] == 'retrain':
        train(retrain=True)

    elif sys.argv[1] == 'evaluate':
        evaluate('training')
        evaluate('validation')

    elif sys.argv[1] == 'test':
        test()
        evaluate('validation')

    else:
        print('Command not found')
import numpy as np
import sys
import torch
from torchvision import transforms, models
from reader import readShortVideo, getVideoList
from RNN import RNN_predictor

## parameters ##
vdo_dir = sys.argv[1]
csv_file = sys.argv[2]
txt_dest = sys.argv[3]

batch_size = 20
model_name = 'rnn.pkl'

def get_features():
    extractor = models.vgg16(pretrained= True).features.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    csv = getVideoList(csv_file)
    
    vdo_num = len(csv['Video_index'])
    frames_num = 10 ## number of frames per clip 
    vdo_features = torch.zeros((vdo_num,frames_num,512*3*5),dtype= torch.float)

    for clip_id in range(vdo_num): 

        frames = readShortVideo(vdo_dir,csv['Video_category'][clip_id],csv['Video_name'][clip_id],2,0.5)
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
        
        frame_features= extractor(frame_sample)  #(frames_num,512,3,5)
            
        frame_features = frame_features.view(frames_num,-1)  ## in shape of (frames_num,512*3*5)
        
        vdo_features[clip_id] = frame_features.detach().cpu()

    print('Finished extracting features')
    return vdo_features

if __name__ == '__main__':
    print('==========  HW5 p2 =============')
    
    vdo_features = get_features()
    vdo_num = vdo_features.shape[0]
    # now get features (vdo_num, 2048,7,10)
    
    txt = open(txt_dest+'p2_result.txt','w')
    predictor = RNN_predictor().cuda()
    predictor.load_state_dict(torch.load(model_name))
    result = torch.zeros((vdo_num ,))

    batch_id = 0
    while  batch_id*batch_size < vdo_num :
        start = batch_id*batch_size 
        end = vdo_num  if (batch_id+1)*batch_size > vdo_num  else (batch_id+1)*batch_size
        in_batch = vdo_features[start:end].cuda()
        out_batch= predictor(in_batch)
        
        label_batch = torch.max(out_batch,1)[1]
        result[start:end] = label_batch
        batch_id += 1
    # now get pridicting result (vdo_num,)
    for i in range(vdo_num):
        txt.write(str(int(result[i]))+'\n')

    txt.close()
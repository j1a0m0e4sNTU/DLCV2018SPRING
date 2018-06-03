import numpy as np
import sys
import torch
from torchvision import transforms, models
from reader import readShortVideo, getVideoList
from CNN import ActionPredictor

## parameters ##
vdo_dir = sys.argv[1]
csv_file = sys.argv[2]
txt_dest = sys.argv[3]

batch_size = 20
model_name = 'cnn.pkl'

def get_features():
    extractor = models.vgg16(pretrained= True).features.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    csv = getVideoList(csv_file)
    
    vdo_num = len(csv['Video_index'])
    vdo_features = torch.zeros((vdo_num,2048,7,10),dtype= torch.float)
    for clip_id in range(vdo_num): #
        frames = readShortVideo(vdo_dir,csv['Video_category'][clip_id],csv['Video_name'][clip_id],5)
        
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
        
        vdo_features[clip_id] = frame_features.detach().cpu()
    
    print('finish extracting features')
    return vdo_features

if __name__ == '__main__':
    print('==========  HW5 p1 =============')
    
    vdo_features = get_features()
    vdo_num = vdo_features.shape[0]
    # now get features (vdo_num, 2048,7,10)
    
    txt = open(txt_dest+'p1_valid.txt','w')
    predictor = ActionPredictor().cuda()
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
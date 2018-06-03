import numpy as np
from matplotlib import pyplot as plt
from reader import getVideoList
import skimage.draw as draw
import matplotlib.patches as patches

def plot_learning_curve():
    loss_file = open('rnn_seq.txt')
    loss_file = list(loss_file)
    loss = []

    for line in loss_file:
        if line.startswith('Epoch'):
            idx = line.find('=')
            loss.append(float(line[idx+2:-1]))

    loss = np.array(loss)
    time = np.arange(loss.shape[0])
    plt.title('RNN from sequence to sequence')
    plt.xlabel('time'); plt.ylabel('loss')
    plt.plot(time,loss)
    
    plt.savefig('rnn_lc.jpg')  
    plt.show()



def ComputeAcc():
    
    csv = getVideoList('data/TrimmedVideos/label/gt_valid.csv')
    colomn = np.array(csv['Action_labels'])
    gt = np.zeros((len(colomn),))
    for i in range(11):
        gt[colomn == str(i)] = i
    '''
    gt_file = open('data/FullLengthVideos/labels/valid/OP01-R03-BaconAndEggs.txt')
    gt_file = np.array(list(gt_file))
    gt = np.zeros((gt_file.shape[0],))
    '''
    result_file = open('test/p1_valid.txt')
    result_file = np.array(list(result_file))
    result = np.zeros((result_file.shape[0],))
    
    for i in range(11):
        result[ result_file == str(i)+'\n'] = i
        #gt[ gt_file == str(i)+'\n'] = i
    print('Acc:',gt[ result == gt].shape[0]/gt.shape[0])

def plot():
    gt_file = open('data/FullLengthVideos/labels/valid/OP01-R03-BaconAndEggs.txt')
    gt_file = np.array(list(gt_file))
    gt = np.zeros((gt_file.shape[0],))
    
    result_file = open('test/OP01-R03-BaconAndEggs.txt')
    result_file = np.array(list(result_file))
    result = np.zeros((result_file.shape[0],))
    
    for i in range(11):
        result[ result_file == str(i)+'\n'] = i
        gt[ gt_file == str(i)+'\n'] = i
    
    brick_w = 0.1
    brick_h = 1
    length = result.shape[0]
    for i in range(length):
        if   result[i] == 0: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c='r')
        elif result[i] == 1: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(0.5,0,0))
        elif result[i] == 2: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(0,0.5,0))
        elif result[i] == 3: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(0,0,0.5))
        elif result[i] == 4: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(1,0,0))
        elif result[i] == 5: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(0,1,0))
        elif result[i] == 6: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(0,0,1))
        elif result[i] == 7: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(1,1,0))
        elif result[i] == 8: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(0,1,1))
        elif result[i] == 9: plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(1,0,1))
        else :              plt.fill((i*brick_w,i*brick_w,(i+1)*brick_w,(i+1)*brick_w),(0,brick_h,brick_h,0),c=(0,1,0))
    
    plt.axis('off')
    plt.savefig('result_label.png')




  
def test():
    plt.fill((0,0,1,1),(0,1,1,40),c=(0,0.5,1))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print(' ---------  helper ----------')
    ComputeAcc()
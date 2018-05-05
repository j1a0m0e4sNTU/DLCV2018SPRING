import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import time
import  matplotlib.image as mpimg

train_dir = "hw3-train-validation/train/"
valid_dir = "hw3-train-validation/validation/"

train_size = 2313
valid_size = 257
rgb = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,1],
    [1,1,0],
    [1,1,1]
],dtype=np.float32)

category = np.array([
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,1,0],
    [0,0,0,0,1,0,0],
    [0,0,0,1,0,0,0],
    [0,0,1,0,0,0,0],
    [0,1,0,0,0,0,0],
    [1,0,0,0,0,0,0]
],dtype=np.uint8)
def category_id(a):
    i = 0
    if a[0] == 1: i = np.add(i,3)
    if a[1] == 1: i = np.add(i,2)
    if a[2] == 1: i = np.add(i,1)
    return i
    
img_id = []
for i in range(train_size):
    if i<10 : img_id.append('000'+str(i))
    elif 10<=i<100: img_id.append('00'+str(i))
    elif 100<=i<1000: img_id.append('0'+str(i))
    else : img_id.append(str(i))

def image_transform(img): #(,,3)  -->  (,,7)
    trans = np.ones((512,512,7),dtype=np.uint8)
    for w in range(512):
        for h in range(512):
            trans[w][h] = category[category_id(img[w][h])]
    return trans

def image_recons(img):  #(,,7) ---> (,,3)
    trans = np.ones((512,512,3),dtype=np.float32)
    for w in range(512):
        for h in range(512):
            trans[w][h] = rgb[6-np.argmax(img[w][h])]
    return trans

def reconstruct(predict): # (,,7)to (,,3)
    color = np.empty((512,512))
    rgb = np.empty((512,512,3),dtype=np.float32)
    for i in range(512):
        for j in range(512):
            color[i][j] = np.argmax(predict[i][j])
    rgb[ color == 0 ] = [1,1,1]
    rgb[ color == 1 ] = [1,1,0]
    rgb[ color == 2 ] = [1,0,1]
    rgb[ color == 3 ] = [0,1,1]
    rgb[ color == 4 ] = [0,1,0]
    rgb[ color == 5 ] = [0,0,1]
    rgb[ color == 6 ] = [0,0,0]
    return rgb

def test():
    x = mpimg.imread(train_dir+img_id[10]+'_sat.jpg').reshape((1,512,512,3))
    y = mpimg.imread(train_dir+img_id[10]+'_mask.png').reshape((1,512,512,3))
    print(x)
    print(y)
    print(sys.getsizeof(x),sys.getsizeof(y))
    

def test2():
    img = mpimg.imread(train_dir+img_id[5]+'_mask.png')
    x = image_transform(img) 
    y = reconstruct(x)
    if np.any(img-y)==False:
        print("Same")
    plt.imshow(img);plt.show()
    plt.imshow(y);plt.show()

    
def save_train1():
    #train_x = mpimg.imread(train_dir+img_id[0]+'_sat.jpg').reshape((1,512,512,3))
    train_y = image_transform(mpimg.imread(train_dir+img_id[0]+'_mask.png')).reshape((1,512,512,8))
    for i in range(1,1200):
        #train_x = np.vstack((train_x,mpimg.imread(train_dir+img_id[i]+'_sat.jpg').reshape((1,512,512,3))))
        train_y = np.vstack((train_y,image_transform(mpimg.imread(train_dir+img_id[i]+'_mask.png')).reshape((1,512,512,8))))
        if i%50 ==0: print(i,"train images finished")

    #np.save("train_x1.npy",train_x)
    np.save("train_y1.npy",train_y)
    print("*****train_1 saved")

def save_train2():
    #train_x = mpimg.imread(train_dir+img_id[1200]+'_sat.jpg').reshape((1,512,512,3))
    train_y = image_transform(mpimg.imread(train_dir+img_id[1200]+'_mask.png')).reshape((1,512,512,8))
    for i in range(1200,train_size):
        #train_x = np.vstack((train_x,mpimg.imread(train_dir+img_id[i]+'_sat.jpg').reshape((1,512,512,3))))
        train_y = np.vstack((train_y,image_transform(mpimg.imread(train_dir+img_id[i]+'_mask.png')).reshape((1,512,512,8))))
        if i%50 ==0: print(i,"train images finished")

    #np.save("train_x2.npy",train_x)
    np.save("train_y2.npy",train_y)
    print("*****train_2 saved")

def save_valid():
    #valid_x = mpimg.imread(valid_dir+img_id[0]+'_sat.jpg').reshape((1,512,512,3))
    valid_y = image_transform(mpimg.imread(valid_dir+img_id[0]+'_mask.png')).reshape((1,512,512,8))
    for i in range(1,valid_size):
        #valid_x = np.vstack((valid_x,mpimg.imread(valid_dir+img_id[i]+'_sat.jpg').reshape((1,512,512,3))))
        valid_y = np.vstack((valid_y,image_transform(mpimg.imread(valid_dir+img_id[i]+'_mask.png')).reshape((1,512,512,8))))
        if i%50 ==0: print(i,"valid images finished")

    #np.save("valid_x.npy",valid_x)
    np.save("valid_y.npy",valid_y)
    print("*****valid saved")
    
def test3():
    im = image_transform(mpimg.imread(train_dir+img_id[5]+'_mask.png'))
    train_y = np.load("train_y1.npy")
    train_y = np.stack((train_y[:,:,:,0],train_y[:,:,:,1],train_y[:,:,:,2],train_y[:,:,:,4],train_y[:,:,:,5],train_y[:,:,:,6],train_y[:,:,:,7]),axis=3)
    if np.any(im-train_y[5]) == False:
        print("Equal!")
    else:
        print("Not equal")

if __name__ == "__main__":
    print("=========main==========")
    test2()

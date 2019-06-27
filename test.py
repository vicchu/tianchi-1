import torch
from torch.utils import data
from model import Unet
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from linknet import LinkNet

torch.backends.cudnn.benchmark=True
torch.manual_seed(10)
file_path='/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/train_1.npy'
model_path='/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/weight/linknet_640.pth' #.pth'

unet=LinkNet()
unet.load_state_dict(torch.load(model_path)['net'])
unet.cuda()
unet.eval()

# - - - - ;

print('loading data ...')
file_=np.load(file_path)[-5000::,0:5000] 
print(file_.dtype)
print('done ...')
h,w,_ = file_.shape
img_size = 640
img_stride = int(img_size/2)
h_ = h// img_stride - 1
w_ = w// img_stride - 1
print('new data ...')
pred=np.zeros((4,h,w),np.float32) 
pred_num=np.zeros((h,w),np.float32) 

with torch.no_grad():
	print('loop 1  ...')
	for index in range(h_*w_):
	    i = index//w_
	    j = index%w_
	    img = file_[i*img_stride:i*img_stride+img_size,j*img_stride:j*img_stride+img_size].copy()
	    img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	    img=(img.float()/255-0.5)/0.5
	    seg=unet(img.float().cuda())[0].data.cpu().numpy()
	    pred[:,i*img_stride:i*img_stride+img_size,j*img_stride:j*img_stride+img_size]+=seg
	    pred_num[i*img_stride:i*img_stride+img_size,j*img_stride:j*img_stride+img_size]+=1
	print('loop 2  ...')    
	for index in range(h_):
	    img = file_[index*img_stride:index*img_stride+img_size,w-img_size:w].copy()
	    img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	    img=(img.float()/255-0.5)/0.5
	    seg=unet(img.float().cuda())[0].data.cpu().numpy()
	    pred[:,index*img_stride:index*img_stride+img_size,w-img_size:w]+=seg
	    pred_num[index*img_stride:index*img_stride+img_size,w-img_size:w]+=1
	print('loop 3  ...')
	for index in range(w_):
	    img = file_[h-img_size:h,index*img_stride:index*img_stride+img_size].copy()
	    img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	    img=(img.float()/255-0.5)/0.5
	    seg=unet(img.float().cuda())[0].data.cpu().numpy()
	    pred[:,h-img_size:h,index*img_stride:index*img_stride+img_size]+=seg
	    pred_num[h-img_size:h,index*img_stride:index*img_stride+img_size]+=1
	print('loop 4  ...')    
	img = file_[h-img_size:h,w-img_size:w].copy()
	img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	img=(img.float()/255-0.5)/0.5
	seg=unet(img.float().cuda())[0].data.cpu().numpy()
	pred[:,h-img_size:h,w-img_size:w]+=seg
	pred_num[h-img_size:h,w-img_size:w]+=1	    
	print('done')


    
pred/=pred_num
pred = np.argmax(pred, axis=0)
print(pred.shape)
del file_,img,seg,pred_num
cv2.imwrite('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/pred/image_1_predict.png', pred)
#pred_num[pred_num==2]=50
#pred_num[pred_num==3]=100
#pred_num[pred_num==4]=150
#cv2.imwrite('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/pred/num.png', pred_num)
    
#print('done')




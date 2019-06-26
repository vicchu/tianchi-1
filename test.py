import torch
from torch.utils import data
from model import Unet
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark=True
torch.manual_seed(10)
file_path='/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/test_4.npy'
model_path='/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/weight/unet_256_noaug.pth' #.pth'

unet=Unet()
unet.load_state_dict(torch.load(model_path))
unet.cuda()
unet.eval()

# - - - - ;

print('loading data ...')
file_=np.load(file_path)#[-10000::,0:10000] 
print(file_.dtype)
print('done ...')
h,w,_ = file_.shape
h_ = h// 128 - 1
w_ = w// 128 - 1
print('new data ...')
pred=np.zeros((4,h,w),np.float32) 
pred_num=np.zeros((h,w),np.float32) 

with torch.no_grad():
	print('loop 1  ...')
	for index in range(h_*w_):
	    i = index//w_
	    j = index%w_
	    img = file_[i*128:i*128+256,j*128:j*128+256].copy()
	    img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	    img=(img.float()/255-0.5)/0.5
	    seg=unet(img.float().cuda())[0].data.cpu().numpy()
	    pred[:,i*128:i*128+256,j*128:j*128+256]+=seg
	    pred_num[i*128:i*128+256,j*128:j*128+256]+=1
	print('loop 2  ...')    
	for index in range(h_):
	    img = file_[index*128:index*128+256,w-256:w].copy()
	    img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	    img=(img.float()/255-0.5)/0.5
	    seg=unet(img.float().cuda())[0].data.cpu().numpy()
	    pred[:,index*128:index*128+256,w-256:w]+=seg
	    pred_num[index*128:index*128+256,w-256:w]+=1
	print('loop 3  ...')
	for index in range(w_):
	    img = file_[h-256:h,index*128:index*128+256].copy()
	    img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	    img=(img.float()/255-0.5)/0.5
	    seg=unet(img.float().cuda())[0].data.cpu().numpy()
	    pred[:,h-256:h,index*128:index*128+256]+=seg
	    pred_num[h-256:h,index*128:index*128+256]+=1
	print('loop 4  ...')    
	img = file_[h-256:h,w-256:w].copy()
	img = torch.from_numpy(img.transpose((2,0,1))[np.newaxis,:])
	img=(img.float()/255-0.5)/0.5
	seg=unet(img.float().cuda())[0].data.cpu().numpy()
	pred[:,h-256:h,w-256:w]+=seg
	pred_num[h-256:h,w-256:w]+=1	    
	print('done')


    
pred/=pred_num
pred = np.argmax(pred, axis=0)
print(pred.shape)
del file_,img,seg,pred_num
cv2.imwrite('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/pred/image_4_predict.png', pred)
#pred_num[pred_num==2]=50
#pred_num[pred_num==3]=100
#pred_num[pred_num==4]=150
#cv2.imwrite('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/pred/num.png', pred_num)
    
#print('done')




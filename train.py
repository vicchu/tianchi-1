import torch
from torch.utils import data
from model import Unet
from dataset import Dataset_map
from utils import seed_torch,save_traindata
from loss import dice_loss
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from linknet import LinkNet
import numpy as np

seed_torch()

print('*******************train_unet*******************')
file_path='/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/train_1.npy'
model_save='/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/weight/linknet_640.pth'#unet_512_noaug.pth' #.pth'
train_data=Dataset_map(file_path)
batch_size=64
train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,drop_last=True,num_workers=4)
unet=LinkNet()
checkpoint = torch.load('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/weight/linknet_640.pth')
unet.load_state_dict(checkpoint['net'])
optimizer=torch.optim.Adam(unet.parameters(),lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer'])
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
unet.cuda()
unet.train()
EPOCH=100#120
TRAIN_LOSS=[]
print('start')
cal_loss = nn.CrossEntropyLoss()
epoch_save = checkpoint['epoch'] + 1
del checkpoint
for epoch in range(epoch_save,EPOCH):
    batch_loss=0
   # random.seed(epoch)
    for i,(img,label,h,w) in enumerate(train_loader):
#         if i==0:  #save train samples
#             save_traindata(img,label)
#             print(epoch,'****',h)
        seg=unet(img.float().cuda())
        loss=cal_loss(seg.view(img.size(0),4),label.view(img.size(0)).long().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
    batch_loss/=EPOCH
    TRAIN_LOSS.append(batch_loss)
    x=[i for i in range(epoch+1)]
    plt.plot(TRAIN_LOSS,label='train loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/weight/train.png',format='png')
    plt.close()

    print('EPOCH %d : train_score = %.4f '%(epoch,batch_loss))
    if 1:#epoch%5==0 or epoch==EPOCH-1:
     checkpoint = {'net':unet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'model':'linknet_noaug','size':640}
     torch.save(checkpoint,model_save) #model_save.replace('30',str(epoch))



import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from linknet import LinkNet
import numpy as np
import torchvision.utils as vutils
def save_traindata(img,label):
    #save train samples
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(img.cuda(), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('train_data.png',format='png')
    plt.close()
    
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid((((label.float().unsqueeze(1)/3.)-0.5)/0.5).cuda(), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig('train_label.png',format='png')
    plt.close()
    
def label2onehot( labels, dim=4): # labels (batch,h,w)
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim , labels.size(1),labels.size(2))  ###可以优化
    for k in range(batch_size):
        for i in range(labels.size(1)):
            for j in range(labels.size(2)):
                out[k, labels[k][i][j].long() , i, j] = 1
    return out
def seed_torch(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':
    torch.manual_seed(1)
    x = torch.rand((128, 128))
    label2onehot

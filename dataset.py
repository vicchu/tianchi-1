from torch.utils import data
import numpy as np
import torch
import random
class Dataset_map(data.Dataset):
    def __init__(self,file):
        self.file=np.load(file) 
        file_seg=file.replace('train_1','train_1_label')
        self.label=np.load(file_seg)
        self.h,self.w,_ = self.file.shape
        
#         self.file2=np.load(file.replace('train_1','train_2')) 
#         file_seg2=file.replace('train_1','train_2_label')
#         self.label2=np.load(file_seg2)
#         self.h2,self.w2,_ = self.file2.shape
        self.size = 640
        self.len1 = int(self.h*self.w/self.size/self.size )
        print('len1',self.len1)
#         self.len2 = int(self.h2*self.w2/self.size/self.size )
#         print('len2',self.len2)
    def __getitem__(self, index):
        if index < 2*self.len1:
           #fix bug :  use random instead of np.random if num_works>1
            h = random.randint(0,self.h-self.size)
            w = random.randint(0,self.w-self.size)
            img = self.file[h:h+self.size,w:w+self.size]
            while (img.max()==0):
                h = random.randint(0,self.h-self.size)
                w = random.randint(0,self.w-self.size)
                img = self.file[h:h+self.size,w:w+self.size]
#             print('**************',h,w)
            label = self.label[h:h+self.size,w:w+self.size]
            img = torch.from_numpy(img.transpose((2,0,1)))
            label = torch.from_numpy(label)
            img=(img.float()/255-0.5)/0.5
        else:
            h2 = random.randint(0,self.h2-self.size)
            w2 = random.randint(0,self.w2-self.size)
            img = self.file2[h2:h2+self.size,w2:w2+self.size]
            while (img.max()==0):
                h2 = random.randint(0,self.h2-self.size)
                w2 = random.randint(0,self.w2-self.size)
                img = self.file2[h2:h2+self.size,w2:w2+self.size]
            label = self.label2[h2:h2+self.size,w2:w2+self.size]
            img = torch.from_numpy(img.transpose((2,0,1)))
            label = torch.from_numpy(label)
            img=(img.float()/255-0.5)/0.5
            
        return img,label,h,w
    def __len__(self):
        return 1000#2*(self.len1) #+self.len2
    
class Dataset_test(data.Dataset):
    def __init__(self,file):
        self.size = 640
        self.file=np.load(file) 
        self.h,self.w,_ = self.file.shape
        self.h_ = self.h//self.size
        self.w_ = self.w//self.size

    def __getitem__(self, index):
        i = index//self.w_
        j = index%self.w
        img = self.file[i*self.size:(i+1)*self.size,j:(j+1)*self.size]
        img = torch.from_numpy(img.transpose((2,0,1)))
        img=(img.float()/255-0.5)/0.5
        return img
    def __len__(self):
        return int(self.h_*self.w_) #+self.h_+self.w_+1

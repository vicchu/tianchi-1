from torch.utils import data
import numpy as np
import torch
class Dataset_map(data.Dataset):
    def __init__(self,file):
        self.file=np.load(file) 
        file_seg=file.replace('train_1','train_1_label')
        self.label=np.load(file_seg)
        self.h,self.w,_ = self.file.shape
        
        self.file2=np.load(file.replace('train_1','train_2')) 
        file_seg2=file.replace('train_1','train_2_label')
        self.label2=np.load(file_seg2)
        self.h2,self.w2,_ = self.file2.shape
        self.size = 512
        self.len1 = int(self.h*self.w/self.size/self.size )
        self.len2 = int(self.h2*self.w2/self.size/self.size )
        print('len1',self.len1)
        print('len2',self.len2)
    def __getitem__(self, index):
#         print('in')
#         print('index',index)
        if index < self.len1:
            h = np.random.randint(0,self.h-self.size)
            w = np.random.randint(0,self.w-self.size)
#             print('(',h,w,')')
            img = self.file[h:h+self.size,w:w+self.size]
    #         print('---')
            while (img.max()==0):
    #             print('777')
                h = np.random.randint(0,self.h-self.size)
                w = np.random.randint(0,self.w-self.size)
                img = self.file[h:h+self.size,w:w+self.size]
    #         print('ffff',h,w)
            label = self.label[h:h+self.size,w:w+self.size]
    #         print('22222')
            img = torch.from_numpy(img.transpose((2,0,1)))
            label = torch.from_numpy(label)
    #         print('ff')
            img=(img.float()/255-0.5)/0.5
        else:
            h2 = np.random.randint(0,self.h2-self.size)
            w2 = np.random.randint(0,self.w2-self.size)
            img = self.file2[h2:h2+self.size,w2:w2+self.size]
    #         print('---')
            while (img.max()==0):
    #             print('777')
                h2 = np.random.randint(0,self.h2-self.size)
                w2 = np.random.randint(0,self.w2-self.size)
                img = self.file2[h2:h2+self.size,w2:w2+self.size]
    #         print('ffff',h,w)
            label = self.label2[h2:h2+self.size,w2:w2+self.size]
    #         print('22222')
            img = torch.from_numpy(img.transpose((2,0,1)))
            label = torch.from_numpy(label)
    #         print('ff')
            img=(img.float()/255-0.5)/0.5
            
        return img,label
    def __len__(self):
        return self.len1+self.len2
    
class Dataset_test(data.Dataset):
    def __init__(self,file):
        self.file=np.load(file) 
        self.h,self.w,_ = self.file.shape
        self.h_ = self.h//256
        self.w_ = self.w//256

    def __getitem__(self, index):
        i = index//self.w_
        j = index%self.w
        img = self.file[i*256:(i+1)*256,j:(j+1)*256]
        img = torch.from_numpy(img.transpose((2,0,1)))
        img=(img.float()/255-0.5)/0.5
        return img
    def __len__(self):
        return int(self.h_*self.w_) #+self.h_+self.w_+1
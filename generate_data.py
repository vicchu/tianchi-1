# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# import numpy as np

# Image.MAX_IMAGE_PIXELS = 10**12
# img = Image.open('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/jingwei_round1_train_20190619/image_1.png')   # 注意修改img路径
# img = np.asarray(img.convert("RGB"))
# # print('finish loading img')
# # label = Image.open('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/jingwei_round1_train_20190619/image_1_label.png')
# # label = np.asarray(label)
# # print('finish loading label')
# # data=[]
# # data.append(img)
# # data.append(label)
# # print(data[0].shape)
# # print(data[1].shape)
# # print('finish saving label')
# print(img.shape)
# np.save('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/train_1.npy',data)
# print('finish')

####
# a=np.load('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/train_1.npy')
# print('finish')
# print(a[0].shape)
# print(a[1].shape)

import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 10**12
img = Image.open('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/jingwei_round1_train_20190619/image_2.png')   # 注意修改img路径
img = img.convert("RGB")
img = np.asarray(img)
np.save('/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/tianchi/data/train_2.npy',img)



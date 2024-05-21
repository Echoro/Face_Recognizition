#-*- coding: utf-8 -*-

import cv2
import sys
import gc
#from AlexNet2 import Model


import matplotlib.pyplot as plt

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

# RTSP URL，替换为您的摄像头地址
rtsp_url = 'rtsp://10.134.142.225:8554/mystream'
rtsp_url = 0  #本机摄像头


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))




#最后剪裁的图片大小,为了和之前的图片处理过程适配
size_m = 160
size_n = 160
image_w = 160 #图片的宽to_categorical
image_h = 160 #图片的高
num_classes = 4 #标签数量

with open(r"../FaceDataAndAlgorithm/Lables.txt","r") as f:
    lines = f.readlines()
num_classes = int(lines[-1].strip().split(";")[-1]) + 1



if len(sys.argv) != 2:
    print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    print(sys.argv)



resnet = InceptionResnetV1(
classify=True,
pretrained='vggface2',
num_classes=num_classes #len(dataset.class_to_idx)  #####3
).to(device)


# 加载模型
model = resnet  # 这里需要你的模型定义
model.load_state_dict(torch.load(r"..\checkpoint\Resnet_Face_Recognition.pth"))
model.eval()
model = model.to(device)
#框住人脸的矩形边框颜色       
color = (0, 255, 0)




# 创建VideoCapture对象
cap = cv2.VideoCapture(rtsp_url)

#人脸识别分类器本地存储路径
cascade_path = r'..\haarcascade_frontalface_default.xml'         ######?



# frame = cv2.imread("C:/Users/77996/Desktop/CNN-FaceRec-keras-master/CNN-FaceRec-keras-master/data/test/55.jpg", cv2.IMREAD_COLOR)


# #图像灰化，降低计算复杂度
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #使用人脸识别分类器，读入分类器
# cascade = cv2.CascadeClassifier(cascade_path)                ###########
# #print("Is cascade empty?", cascade.empty())

# #利用分类器识别出哪个区域为人脸
# faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (20, 20))        #####
# if len(faceRects) > 0:                 
#     for faceRect in faceRects: 
#         x, y, w, h = faceRect

#         #截取脸部图像提交给模型识别这是谁
#         #img = frame[y - 10: y + h + 10, x - 10: x + w + 10]

        
#         margin_x = max(0, int(0.1 * w))  # 根据脸部宽度调整X方向的边界
#         margin_y = max(0, int(0.1 * h))  # 根据脸部高度调整Y方向的边界

#         img = frame[y - margin_y: y + h + margin_y, x - margin_x: x + w + margin_x] ######

#         if img is not None and img.size > 0:
#             resized_frame = cv2.resize(img, (size_m, size_n))
#             # 其他处理...
#         else:
#             print("Error: Unable to resize. Empty or invalid image.")



#         # 将OpenCV图像格式转换为Pillow图像格式
            
#         img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

# #img = Image.open("C:/Users/77996/Desktop/CNN-FaceRec-keras-master/CNN-FaceRec-keras-master/data/test/微信图片_20240122182839.jpg")


# img = img.resize((image_w, image_h), Image.ANTIALIAS)
# img_arr = np.array(img,dtype = np.float32)

# #img_arr = img_arr / 255   ######调用trans函数
# img_tensor = torch.from_numpy(img_arr)
# img_tensor = img_tensor.unsqueeze(0)  # 添加一个额外的维度
# img_tensor = img_tensor.permute(0, 3, 1, 2)
# img_tensor=img_tensor.float()
# img_tensor = fixed_image_standardization(img_tensor)

# img_tensor=img_tensor.to(device)

# with torch.no_grad():
#     prediction = model(img_tensor)

# pred = prediction.cpu().numpy()
# # 计算softmax
# exp = np.exp(pred)
# pred = exp / np.sum(exp)
# max_probability = np.max(pred)
# print(max_probability)
# max_index = np.argmax(pred)
# print(max_index)

#循环检测识别人脸
while True:
    _, frame = cap.read()   #读取一帧视频
    #图像灰化，降低计算复杂度
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #使用人脸识别分类器，读入分类器
    cascade = cv2.CascadeClassifier(cascade_path)                ###########
    #print("Is cascade empty?", cascade.empty())

    #利用分类器识别出哪个区域为人脸
    faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (20, 20))        #####
    if len(faceRects) > 0:                 
        for faceRect in faceRects: 
            x, y, w, h = faceRect

            #截取脸部图像提交给模型识别这是谁
            #img = frame[y - 10: y + h + 10, x - 10: x + w + 10]

            
            margin_x = max(0, int(0.1 * w))  # 根据脸部宽度调整X方向的边界
            margin_y = max(0, int(0.1 * h))  # 根据脸部高度调整Y方向的边界

            img = frame[y - margin_y: y + h + margin_y, x - margin_x: x + w + margin_x] ######

            if img is not None and img.size > 0:
                resized_frame = cv2.resize(img, (size_m, size_n))
                # 其他处理...
            else:
                print("Error: Unable to resize. Empty or invalid image.")



            # 将OpenCV图像格式转换为Pillow图像格式
                
            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

            img = img.resize((image_w, image_h), Image.ANTIALIAS)
            img_arr = np.array(img,dtype = np.float32)

            #img_arr = img_arr / 255   ######调用trans函数
            img_tensor = torch.from_numpy(img_arr)
            img_tensor = img_tensor.unsqueeze(0)  # 添加一个额外的维度
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            img_tensor=img_tensor.float()
            img_tensor = fixed_image_standardization(img_tensor)
      
            img_tensor=img_tensor.to(device)
     
            with torch.no_grad():
                prediction = model(img_tensor)

            pred = prediction.cpu().numpy()
            # 计算softmax
            exp = np.exp(pred)
            pred = exp / np.sum(exp)
            max_probability = np.max(pred)
            print(max_probability)
            max_index = np.argmax(pred)
            print(max_index)

            
                    
            
            
            if max_probability < 0.9:
                cv2.rectangle(frame, (x - margin_x, y - margin_y), (x + w + margin_x, y + h + margin_y), color, thickness = 2)
                

                #文字提示是谁
                cv2.putText(frame,"don't find the person", 
                            (x + 30, y + 30),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体
                            1,                                     #字号
                            (255,0,255),                           #颜色
                            2)                                     #字的线宽
            else :
                n = len(lines)
                i = 0

                while i<n:

                # 获取一个BatchSize大小的数据
                #for b in range(BatchSize):
                    if i==0:
                        np.random.shuffle(lines)
                    #name = lines[i].split(';')[0]
                    name = lines[i].split(';')[1]

                    #label_name = (lines[i].split(';')[1]).strip('/n')
                    label_name = (lines[i].split(';')[2]).strip('\n')

                    file_name = lines[i].split(';')[0]

                    if int(label_name) == max_index:
                        print(file_name)
                        cv2.rectangle(frame, (x - margin_x, y - margin_y), (x + w + margin_x, y + h + margin_y), color, thickness = 2)

                        #文字提示是谁
                        cv2.putText(frame,file_name, 
                                    (x + 30, y + 30),                      #坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                    1,                                     #字号
                                    (255,0,255),                           #颜色
                                    2)                                     #字的线宽
                        break

    cv2.imshow("Recognise myself", frame)

    #等待10毫秒看是否有按键输入
    k = cv2.waitKey(10)
    #如果输入q则退出循环
    if k & 0xFF == ord('q'):
        break

#释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()
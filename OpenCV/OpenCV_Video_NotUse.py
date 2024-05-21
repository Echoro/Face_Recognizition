#-*- coding: utf-8 -*-

import cv2
import sys
import gc
#from AlexNet2 import Model

from tensorflow.keras.regularizers import l2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from PIL import Image



if __name__ == '__main__':

    #最后剪裁的图片大小,为了和之前的图片处理过程适配
    size_m = 48
    size_n = 48
    image_w = 32 #图片的宽to_categorical
    image_h = 32 #图片的高
    num_classes = 434 #标签数量

    with open(r".\data/train.txt","r") as f:
        lines = f.readlines()



    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        print(sys.argv)
        #sys.exit(0)

    #加载模型
    # class AlexNet8(Model):
    #     def __init__(self):
    #         super(AlexNet8, self).__init__()
    #         self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
    #         self.b1 = BatchNormalization()
    #         self.a1 = Activation('relu')
    #         self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

    #         self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
    #         self.b2 = BatchNormalization()
    #         self.a2 = Activation('relu')
    #         self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

    #         self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
    #                         activation='relu')
                            
    #         self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
    #                         activation='relu')
                            
    #         self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
    #                         activation='relu')
    #         self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

    #         self.flatten = Flatten()
    #         self.f1 = Dense(2048, activation='relu')
    #         self.d1 = Dropout(0.5)
    #         self.f2 = Dense(2048, activation='relu')
    #         self.d2 = Dropout(0.5)
    #         self.f3 = Dense(num_class, activation='softmax')

    #     def call(self, x):
    #         x = self.c1(x)
    #         x = self.b1(x)
    #         x = self.a1(x)
    #         x = self.p1(x)

    #         x = self.c2(x)
    #         x = self.b2(x)
    #         x = self.a2(x)
    #         x = self.p2(x)

    #         x = self.c3(x)

    #         x = self.c4(x)

    #         x = self.c5(x)
    #         x = self.p3(x)

    #         x = self.flatten(x)
    #         x = self.f1(x)
    #         x = self.d1(x)
    #         x = self.f2(x)
    #         x = self.d2(x)
    #         y = self.f3(x)
    #         return y


    # model = AlexNet8()
    # model.load_weights('./checkpoint/AlexNet8.ckpt')    
        

    


    # # 加载预训练的ResNet模型
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_w, image_h, 3))

    # # 添加自定义层
    # x = base_model.output
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)  # 添加丢弃层
    # predictions = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)
    # model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)


    # model.load_weights('./REpath/ResNet.ckpt')    

    class ResnetBlock(Model):

        def __init__(self, filters, strides=1, residual_path=False):
            super(ResnetBlock, self).__init__()
            self.filters = filters
            self.strides = strides
            self.residual_path = residual_path

            self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
            self.b1 = BatchNormalization()
            self.a1 = Activation('relu')

            self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
            self.b2 = BatchNormalization()

            # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
            if residual_path:
                self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
                self.down_b1 = BatchNormalization()
            
            self.a2 = Activation('relu')

        def call(self, inputs):
            residual = inputs  # residual等于输入值本身，即residual=x
            # 将输入通过卷积、BN层、激活层，计算F(x)
            x = self.c1(inputs)
            x = self.b1(x)
            x = self.a1(x)

            x = self.c2(x)
            y = self.b2(x)

            if self.residual_path:
                residual = self.down_c1(inputs)
                residual = self.down_b1(residual)

            out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
            return out


    class ResNet18(Model):

        def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
            super(ResNet18, self).__init__()
            self.num_blocks = len(block_list)  # 共有几个block
            self.block_list = block_list
            self.out_filters = initial_filters
            self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
            self.b1 = BatchNormalization()
            self.a1 = Activation('relu')
            self.blocks = tf.keras.models.Sequential()
            # 构建ResNet网络结构
            for block_id in range(len(block_list)):  # 第几个resnet block
                for layer_id in range(block_list[block_id]):  # 第几个卷积层

                    if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                        block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                    else:
                        block = ResnetBlock(self.out_filters, residual_path=False)
                    self.blocks.add(block)  # 将构建好的block加入resnet
                self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
            self.p1 = tf.keras.layers.GlobalAveragePooling2D()
            self.f1 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

        def call(self, inputs):
            x = self.c1(inputs)
            x = self.b1(x)
            x = self.a1(x)
            x = self.blocks(x)
            x = self.p1(x)
            y = self.f1(x)
            return y


    model = ResNet18([2, 2, 2, 2])
    model.load_weights('./checkpoint/ResNet18.ckpt')
    #框住人脸的矩形边框颜色       
    color = (0, 255, 0)

    #捕获指定摄像头的实时视频流
    #cap = cv2.VideoCapture(int(sys.argv[1]))
    
    cap = cv2.VideoCapture(0)

    #人脸识别分类器本地存储路径
    cascade_path = 'C:/Users/77996/Desktop/CNN-FaceRec-keras-master/CNN-FaceRec-keras-master/haarcascade_frontalface_default.xml'         ######?
   # tt = 0
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



                #resized_frame = cv2.resize(img, (32, 32))

                # 将OpenCV图像格式转换为Pillow图像格式
                    
                img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                # if tt < 5 :
                #     img = img = Image.open("C:/Users/77996/Desktop/CNN-FaceRec-keras-master/CNN-FaceRec-keras-master/data/DetectFace/zhaolzy/zhaolzy_"+str(tt)+ ".jpg")
            
            





                img = img.resize((image_w, image_h), Image.ANTIALIAS)
                img_arr = np.array(img,dtype = np.float16)
                #     # 计算均值
                # mean = np.mean(img_arr, axis=(0, 1, 2))

                # # 计算标准差
                # std = np.std(img_arr, axis=(0, 1, 2))

                # # 零均值标准化
                # img_arr = (img_arr - mean) / std
                img_arr = img_arr / 255

                x_predict = img_arr[tf.newaxis, ...]

                result = model.predict(x_predict)
                pred = tf.argmax(result, axis=1)

                pred=np.array(pred)
                max_probability = tf.reduce_max(result, axis=1).numpy()[0]
                
                if max_probability < 0.1:
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

                        if int(label_name) == pred[0]:
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
                # tt = tt +1


                                            
                # else:
                #     pass
            # i = i + 1

        #cv2.imshow("Recognise myself", frame)

        #等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        #如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
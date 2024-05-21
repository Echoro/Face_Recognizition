
"""
Created on Tue May  7 08:38:42 2019
@author: LZY
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import os
 
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
 
dir="../Detected1"
save = "../Enhanced2"
for filename in os.listdir(dir):   
    print(filename)           #listdir的参数是文件夹的路径
    # 生成文件夹路径
    folder_path = os.path.join(save, filename)

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    for imgname in os.listdir(dir+'/'+filename):

        img = load_img(dir+'/'+filename+'/'+imgname)  # 这是一个PIL图像
        x = img_to_array(img)  # 把PIL图像转换成一个numpy数组，形状为(3, 150, 150)
        x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
        # 下面是生产图片的代码
        # 生产的所有图片保存在 `preview/` 目录下
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                save_to_dir=folder_path, 
                                save_prefix='6', 
                                save_format='jpg'):
            i += 1
            if i > 13:
                break  # 否则生成器会退出循环
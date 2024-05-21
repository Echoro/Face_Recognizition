import os

Name_label = [] #姓名标签
path = "../Enhanced2"   #数据集文件路径
dir = os.listdir(path)  #列出所有人

dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

label = 0   #设置计数器

#数据写入
with open("../Lables.txt",'w') as f:
    for name in dir:
        Name_label.append(name)
        print(Name_label[label])
        after_generate = os.listdir(path +'\\'+ name)
        for image in after_generate:
            if image.endswith(".png") or image.endswith(".jpg"):
                #f.write(image + ";" + str(label)+ "\n")
                f.write(name + ";"+image + ";" + str(label)+ "\n")
        label += 1


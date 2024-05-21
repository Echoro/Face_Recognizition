from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

#数据集应该遵循VGGFace2/ImageNet风格的目录布局。将`data_dir`修改为您要微调的数据集所在的位置。

data_dir = "../FaceDataAndAlgorithm/Enhanced2"  #####!!!
batch_size = 60
epochs = 8
workers = 0 if os.name == 'nt' else 8

#判断是否有nvidia GPU可用
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))

# #定义MTCNN模块
# mtcnn = MTCNN(
#     image_size=160, margin=0, min_face_size=20,
#     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#     device=device
# )

# #执行MTCNN人脸检测
# # 迭代DataLoader对象并获取裁剪后的人脸。
dataset = datasets.ImageFolder(data_dir)#, transform=transforms.Resize((512, 512)))
# dataset.samples = [
#     (p, p.replace(data_dir))#, data_dir + '_cropped'))
#         for p, _ in dataset.samples
# ]
        
# loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     collate_fn=training.collate_pil
# )
# for i, (x, y) in enumerate(loader):
#     try:
#         mtcnn(x, save_path=y)
#     except ValueError as e:
#         print(f"Error processing batch {i}: {e}")
#         print(f"Skipping batch {i}.")
#         continue
#     print('\r第 {} 批，共 {} 批'.format(i + 1, len(loader)), end='')


#Remove mtcnn to reduce GPU memory usage
# del mtcnn

# 定义Inception Resnet V1模块
resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(dataset.class_to_idx)
).to(device)


# 定义优化器、调度器、数据集和数据加载器
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir , transform=trans) #+ '_cropped', transform=trans)   ####!!!
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)


# 定义损失和评估函数
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

# 训练模型
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10
try:
    model = resnet  # 这里需要你的模型定义
    model.load_state_dict(torch.load("../checkpoint/Resnet_Face_Recognition.pth"))
    resnet = model
except:


    print('\n\n初始化')
    print('-' * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    epochs = 2
    for epoch in range(epochs):
        print('\n循环 {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    # 在训练循环结束后保存模型
    torch.save(resnet.state_dict(), '../checkpoint/Resnet_Face_Recognition.pth')

    writer.close()
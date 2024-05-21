import os
from PIL import Image

def resize_image(image_path, target_size=(160, 160)):
    """
    调整图像大小为目标尺寸
    
    参数:
    image_path (str): 图像文件路径
    target_size (tuple): 目标尺寸，默认为 (160, 160)
    """
    # 打开图像文件
    with Image.open(image_path) as img:
        # 获取图像的尺寸
        width, height = img.size

        # 如果图像尺寸不符合目标尺寸，则调整大小
        if width != target_size[0] or height != target_size[1]:
            img = img.resize(target_size, Image.ANTIALIAS)

        # 保存调整大小后的图像，覆盖原始文件
        img.save(image_path)
        print(f"Resized {image_path} to {target_size}")

def process_images(directory):
    """
    处理目录中的所有图片文件
    
    参数:
    directory (str): 图像目录路径
    """
    # 遍历目录中的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否为图像文件
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # 拼接文件的完整路径
                image_path = os.path.join(root, file)
                # 调整图像大小
                resize_image(image_path)

# 指定人名目录的父目录路径
parent_directory = r"../Enhanced2"

# 处理所有人名目录下的图片
for person_directory in os.listdir(parent_directory):
    person_directory_path = os.path.join(parent_directory, person_directory)
    if os.path.isdir(person_directory_path):
        process_images(person_directory_path)

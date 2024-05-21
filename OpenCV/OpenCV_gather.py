import cv2
import os

name = "zlzy"

# 创建一个文件夹用于保存图片
output_folder = "C:/Users/77996/Desktop/FACE/NOWuse/RawPictures/newPicture/"+name 
os.makedirs(output_folder, exist_ok=True)

# 初始化摄像头
rtsp_url = 'rtsp://your_camera_address'
rtsp_url = 0 #本机摄像头
cap = cv2.VideoCapture(rtsp_url)

# 设定图像的宽度和高度
cap.set(3, 640)  # 3对应宽度
cap.set(4, 480)  # 4对应高度

# 计数器，用于记录已拍摄的图片数量
image_count = 0
count = 0

# 循环拍摄图片
while 1 : #image_count < 500:
    ret, frame = cap.read()

    # 保存图片
    if image_count%10 == 1:  #实现一tick保存一frame
        image_path = os.path.join(output_folder, f"captured_image_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        count = count + 1 

    # 显示当前帧
    cv2.imshow("Captured Image", frame)

    image_count += 1

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()

print(f"{count} images captured and saved to {output_folder}")

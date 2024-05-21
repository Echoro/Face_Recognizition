@echo off
setlocal enabledelayedexpansion

REM 获取当前批处理文件所在目录
set script_directory=%~dp0



REM 进入 subdirectory1 并运行 script2.py


cd %script_directory%\OpenCV
python OpenCV_gather.py

cd %script_directory%\FaceDataAndAlgorithm\FaceAlgorithm
@REM cd %script_directory%\FaceDataAndAlgorithm\Enhanced2
@REM python fix.py

python findFace.py
python enhance.py

REM 删除文件夹
rd /s /q %script_directory%\FaceDataAndAlgorithm\Detected1
REM 创建新文件夹
mkdir %script_directory%\FaceDataAndAlgorithm\Detected1

python Generate_TrainTXT.py

cd %script_directory%\algorithm
python Train.py

cd %script_directory%\OpenCV
python OpenCV_ResNet_torch.py

endlocal

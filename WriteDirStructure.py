import os

def write_directory_structure(root_dir, output_file):
    with open(output_file, 'w') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            depth = dirpath.count(os.sep)  # 计算当前目录的深度
            indent = ' ' * 4 * depth  # 根据深度设置缩进

            # 写入当前目录
            f.write(f"{indent}+ {os.path.basename(dirpath)}/\n")
            
            if   len(filenames) > 20:
                 f.write(f"{indent}    - the dir has {len(filenames)}pictures\n")
                 continue
            else:
                # 写入当前目录下的文件
                for filename in filenames:
                    f.write(f"{indent}    - {filename}\n")

            # # 写入当前目录下的子文件夹
            # for dirname in dirnames:
            #     f.write(f"{indent}    + {dirname}/\n")

# 要写入目录结构的根目录
root_directory = "C:/Users/77996/Desktop/FACE/NOWuse"

# 要输出的文件名
output_file = 'directory_structure.txt'

# 写入目录结构到文件
write_directory_structure(root_directory, output_file)

print(f"Directory structure has been written to {output_file}.")

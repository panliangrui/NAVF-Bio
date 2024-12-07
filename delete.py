import os
import glob

# 删除指定文件夹中的所有文件
def delete_all_files_in_directory(directory_path):
    # 获取目录下的所有文件路径
    files = glob.glob(os.path.join(directory_path, '*'))
    for file in files:
        try:
            # 删除文件
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

# 示例使用
directory_path = './input'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/FEATURES_DIRECTORY_1024/h5_files'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/FEATURES_DIRECTORY_1024/pt_files'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/FEATURES_DIRECTORY_512/h5_files'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/FEATURES_DIRECTORY_512/pt_files'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/FEATURES_DIRECTORY_256/h5_files'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/FEATURES_DIRECTORY_256/pt_files'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_1024/patches'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_1024/stitches'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_1024/masks'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_512/patches'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_512/stitches'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_512/masks'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_256/patches'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_256/stitches'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './features/RESULTS_DIRECTORY_256/masks'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)

directory_path = './multi_graph_1'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './output'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)
directory_path = './output1'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)

directory_path = './TME'  # 请将此路径替换为您的文件夹路径
delete_all_files_in_directory(directory_path)

import os

# 删除指定的 Excel 文件
def delete_excel_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# 示例使用
file_path = './genetic_analysis.xlsx'  # 请将此路径替换为要删除的 Excel 文件路径
delete_excel_file(file_path)


# import os
#
# def delete_all_files_in_directory(directory):
#     # 遍历目录中的所有文件
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         # 如果是文件，则删除
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#
# # 使用示例
# directory_path = './features/FEATURES_DIRECTORY_1024'
# delete_all_files_in_directory(directory_path)

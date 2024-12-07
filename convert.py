import os
import glob


def convert_kbf_to_svs(kbf_file, svs_file):
    # 这里假设有一个转换函数将kbf文件转换为svs文件
    # 你需要根据实际工具或库实现转换功能
    print(f"Converting {kbf_file} to {svs_file}")
    # 在此处调用实际的转换逻辑
    # 例如，使用某个转换工具或者API：
    # kbf_to_svs_converter.convert(kbf_file, svs_file)


def get_files_in_folder(folder_path):
    # 获取文件夹下所有以 .svs 或 .kbf 结尾的文件
    svs_files = glob.glob(os.path.join(folder_path, "*.svs"))
    kbf_files = glob.glob(os.path.join(folder_path, "*.kbf"))

    return svs_files, kbf_files


def process_files(folder_path):
    # 获取所有svs和kbf文件
    svs_files, kbf_files = get_files_in_folder(folder_path)

    # 对于所有.kbf文件进行转换
    for kbf_file in kbf_files:
        # 定义转换后的svs文件名
        svs_file = os.path.splitext(kbf_file)[0] + ".svs"
        # 执行转换
        convert_kbf_to_svs(kbf_file, svs_file)

    # 输出已经存在的.svs文件
    print(f"Existing .svs files in the folder:")
    for svs_file in svs_files:
        print(svs_file)


if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = "./input"  # 请替换为你自己的文件夹路径
    process_files(folder_path)

# '/home/yyt/Desktop/shanten/archive/2024/2024/2024010100gm-00a9-0000-0d9240dd.mjson'
# '/home/yyt/Desktop/shanten'

import json
import os
import gzip
import sys


def mjson_to_txt(input_path, output_path):
    """
    读取.mjson文件（可能是gzip压缩的），输出为.txt文件
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"✗ 错误：输入文件不存在 '{input_path}'")
            return

        # 尝试以 gzip 方式读取
        try:
            with gzip.open(input_path, 'rt', encoding='utf-8') as f:
                content = f.read()
            print("✓ 检测到 gzip 压缩格式，已自动解压")
        except (gzip.BadGzipFile, OSError):
            # 如果不是 gzip，则按普通文本读取
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print("✓ 检测到普通文本格式")

        # 尝试解析为JSON并格式化
        try:
            data = json.loads(content)
            formatted_content = json.dumps(data, ensure_ascii=False, indent=4)
        except json.JSONDecodeError:
            formatted_content = content  # 保留原始内容

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ 创建输出目录：{output_dir}")

        # 写入.txt文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)

        print(f"✓ 转换成功！")
        print(f"  输入文件：{input_path}")
        print(f"  输出文件：{output_path}")

    except PermissionError:
        print(f"✗ 错误：没有权限访问文件")
    except Exception as e:
        print(f"✗ 错误：{e}")


# ==================== 使用方式 ====================

if __name__ == "__main__":
    # 方式1：直接在代码中指定路径
    input_file = '/home/yyt/Desktop/shanten/archive/2024/2024/2024010100gm-00a9-0000-446cb297.mjson'  # 修改为你的输入路径
    output_file = '/home/yyt/Desktop/shanten/出错2局.txt'  # 修改为你的输出路径
    mjson_to_txt(input_file, output_file)

    # 方式2：从命令行参数获取路径
    # 使用方法：python script.py /path/to/input.mjson /path/to/output.txt
    # if len(sys.argv) == 3:
    #     mjson_to_txt(sys.argv[1], sys.argv[2])
    # else:
    #     print("用法：python script.py <输入路径> <输出路径>")
import subprocess
import os

def delete_output_csv():
    # 检查是否存在 output.csv 文件，如果存在，则删除它
    if os.path.exists('output.csv'):
        os.remove('output.csv')

def run_scripts():
    # 执行1.py
    subprocess.run(['python', '1.py'])

    # 执行2.pyx``
    subprocess.run(['python', '2.py'])

    # 删除 Result.csv 文件
    if os.path.exists('Result.csv'):
        os.remove('Result.csv')

if __name__ == "__main__":
    # 删除 output.csv 文件
    # delete_output_csv()

    # 执行脚本
    run_scripts()

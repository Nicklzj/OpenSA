import tkinter as tk
from tkinter import messagebox
import argparse
parser = argparse.ArgumentParser(description='Your Description Here')

# 添加参数选项
parser.add_argument('--drugname', type=str, default='None',
                    help='Description of  drugname.')
parser.add_argument('--indicators1', type=float, default=1,
                    help='Description of  drugname.')
parser.add_argument('--indicators2', type=float, default=1,
                    help='Description of  drugname.')
parser.add_argument('--indicators3', type=float, default=1,
                    help='Description of  drugname.')
parser.add_argument('--indicators4', type=float, default=1,
                    help='Description of  drugname.')

args = parser.parse_args()
def show_message():
    # 弹出窗口的标题和内容
    title = "检测结果"
    message = args.drugname
    # 显示弹出窗口
    top = tk.Toplevel()
    top.title(title)
    
    # 设置窗口大小
    top.geometry("600x250")
    
    # 创建标签并设置字体和布局
    label = tk.Label(top, text=message, font=('微软雅黑', 20))
    label.pack(expand=True)
    label = tk.Label(top, text="FFT+Cosine similarity: "+str(args.indicators1), font=('微软雅黑', 15))
    label.pack(expand=True)
    label = tk.Label(top, text="SID similarity: "+str(args.indicators2), font=('微软雅黑', 15))
    label.pack(expand=True)
    label = tk.Label(top, text="SAM similarity: "+str(args.indicators3), font=('微软雅黑', 15))
    label.pack(expand=True)
    label = tk.Label(top, text="time : "+str(args.indicators4), font=('微软雅黑', 10))
    label.pack(expand=True)

    # 创建按钮关闭窗口
    # 定义关闭窗口的函数
    def close_window():
        top.destroy()
        root.destroy()  # 关闭主窗口并退出程序
    
    # 创建按钮关闭窗口
    button = tk.Button(top, text="close", command=close_window, font=('微软雅黑', 12))
    button.pack(pady=10)
# 创建主窗口
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 显示消息弹出窗口
show_message()

# 启动tkinter主循环
root.mainloop()


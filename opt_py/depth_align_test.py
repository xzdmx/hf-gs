import numpy as np
import matplotlib.pyplot as plt


def read_and_plot_npy(file_path):
    try:
        data = np.load(file_path)
        if data.ndim == 1:
            # 绘制一维数据的折线图
            plt.plot(data)
            plt.title('Plot of 1D Data from NPY')
            plt.xlabel('Index')
            plt.ylabel('Value')
        elif data.ndim == 2:
            if data.shape[0] == 3 and data.shape[1] == data.shape[2]:
                # 假设是三维图像数据（RGB）
                plt.imshow(data)
                plt.title('Image from NPY')
            else:
                # 绘制二维数据的图像
                plt.imshow(data, cmap='viridis')
                plt.title('Plot of 2D Data from NPY')
                plt.colorbar()
        else:
            print(f"Unsupported data dimension {data.ndim} for plotting.")
        plt.show()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# 替换为你的.npy文件的实际路径
npy_file_path = '/media/junz/4TB-1/ldh/papers/0_Infusion/output/in2n-data/bear/train/ours_1/depth/frame_00033.npy'
read_and_plot_npy(npy_file_path)

npy_file_path = '/media/junz/4TB-1/ldh/papers/0_Infusion/output/in2n-data/bear/depth_completed/depth_completed_frame_00033/frame_00033_depth.npy'
read_and_plot_npy(npy_file_path)

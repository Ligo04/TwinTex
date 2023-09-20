# Multi-Mask RePaint (MMRP)

核心思想是利用扩散模型在采样阶段，对初始输入噪声的高敏感度以间接实现对过拟合效应的抑制。
关键改进有两个：

- 对缺失区域进行分块，各自初始化为相同的高斯噪声
- 对分块的不同区域，初始化以参数不同的高斯噪声

## 预训练模型

```
https://www.dropbox.com/scl/fi/mc3gsvsbxhp27sl0rknbw/ema_0.9999_151161.pt?rlkey=te3n8gxt3op0zkpxltxxnba79&dl=0
```

## 文件及部分代码结构

- `extract_mask.py`：图片预处理
- `mmrp_inpainting.py`： mmrp的主函数
- `inpaint.py`: repainted 
- `conf_mgt/*`：参数文件处理、文件写入
- `confs/*`：参数文件目录
- `data/datasets/*`：ground truth 和掩膜
- `data/pretrained/*`：预训练模型
- `guided_diffusion/*`：扩散模型相关
- `input/*`：输入图片目录
- `log/*`：中间结果目录
- `result/*`：最终结果目录
- `temp`：临时文件目录（转正、裁剪等）

## houdini py环境安装

1. 安装环境

   Windows 按照路径导航到 Houdini 的安装目录，然后进入 Python 文件夹（例如 python39）。如果 pip.py 不存在，则需要先[下载 pip](https://bootstrap.pypa.io/get-pip.py)，然后在此文件夹中打开 Windows 终端并输入命令安装pip。

   ```
   python3.9.exe get-pip.py
   ```

   将项目的`requirements.txt`放到对应的python目录下， 通过`requirements.txt`安装环境

   ```
   python3.9.exe -m pip install -r requirements.txt
   ```
   
   安装pytorch:
   
   ```
   python3.9.exe -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

`inpaint`后在`Results/inpainting`目录下会增加：

```bash
├── ./data/datasets/gt_keep_masks
│   ├── planexxxx_twintex46365/000000.png  # 提取的标识缺失区域的掩膜，黑色为缺失，白色为已知
│   ├── planexxxx_twintex46365_filter_mask/000000.png  # 设置的不需要补全的区域的掩膜
│   ├── planexxxx_twintex46365_sketch1/000000.png # 掩膜分块1
│   ├── planexxxx_twintex46365_sketch2/000000.png # 掩膜分块2
│   ├── planexxxx_twintex46365_sketch3/000000.png # 掩膜分块3

├── ./data/datasets/gts
│   ├── plane298/000000.png  # ground truth，512x512

./confs/
│   ├── plane298_twintex46365.yml
```




# Diffusers

<p align="left">
        <b>简体中文</b> |
</p>

- [FLUX](#jump1)
  - [模型介绍](#模型介绍)
  - [微调](#微调)
    - [环境搭建](#环境搭建)
    - [微调](#jump2)
    - [性能](#性能)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
    - [性能](#jump3)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# FLUX

## 模型介绍

[FLUX.1 dev](https://blackforestlabs.ai/announcing-black-forest-labs/) 是一种基于Rectified Flow Transformers (矫正流) 的生成模型。

- 参考实现：

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=
  ```

## 微调

### 环境搭建

【模型开发时推荐使用配套的环境版本】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          Driver           |         在研版本          |
|         Firmware          |         在研版本          |
|           CANN            |             在研版本             |
|           Torch           |            2.1.0            |
|         Torch_npu         |           2.1.0           |

1. 软件与驱动安装

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install torch-2.1.0*-cp38-cp38m-linux_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 FLUX [GitHub参考实现](https://github.com/huggingface/diffusers) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    cd diffusers
    git checkout 
    cp -r ../MindSpeed-MM/examples/diffusers/flux ./examples/dreambooth
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install e .
    vim examples/dreambooth/requirements_flux.txt #修改torchvision版本：torchvision==0.16.0, torch==2.1.0
    pip install -r examples/dreambooth/requirements_flux.txt # 安装对应依赖
    ```

<a id="jump2"></a>

## 微调

1. 【准备微调数据集】

    - 只包含图片的训练数据集，如非deepspeed脚本使用训练数据集dog:[下载地址](https://huggingface.co/datasets/diffusers/dog-example)，并将dog文件夹转移到examples/dreambooth/目录下

    ```shell
    input_dir="dog" # 数据集路径
    ```

    ```shell
    dog
    ├── alvan-nee-*****.jpeg
    ├── alvan-nee-*****.jpeg
    ```

    > **说明：**
    >该数据集的训练过程脚本只作为一种参考示例。
    >

2. 【配置 FLUX 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[FLUX.1-dev模型](https://huggingface.co/black-forest-labs/FLUX.1-dev) `model_name`模型

    ```bash
    export model_name="black-forest-labs/FLUX.1-dev" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    scripts_path="./flux" # 模型根目录（模型文件夹名称）
    model_name="black-forest-labs/FLUX.1-dev" # 预训练模型路径
    batch_size=16
    max_train_steps=5000
    mixed_precision="bf16" # 混精
    resolution=256
    config_file="pretrain_${mixed_precision}_accelerate_config.yaml"
    ```
    
3. 【修改代码文件】

    在 `src/diffusers/models/embeddings.py` 文件里，在 `class FluxPosEmbed(nn.Module):` 下的 **第760行左右** 找到代码： `freqs_dtype = torch.float32 if is_mps else torch.float64` 进行修改, 请修改为：`freqs_dtype = torch.float32`

    ```shell
    # 修改为freqs_dtype = torch.float32
    vim src/diffusers/models/embeddings.py
    ```

3. 【启动 FLUX 微调脚本】

    本任务主要提供flux_sd3_text2img_dreambooth_flux.sh脚本，该脚本为FLUX微调脚本，支持多卡训练。

    ```shell
    bash flux/sd3_text2img_dreambooth_flux.sh 
    ```

### 性能

#### 吞吐

FLUX 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Flux-全参微调  |   17.08 |     1      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | Flux-全参微调  |  17.51 |     1      | bf16 | 2.1 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**
  
 【运行推理的脚本】

- 单机单卡推理
- 调用推理脚本

  ```shell
  python infer_flux_text2img_bf16.py
  ```

<a id="jump3"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:---:|:---:|:---:|
| 竞品A | 8p |  文生图全参  | - | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |  文生图全参  | - | bf16 | 2.1 | ✔ 

## 引用

### 公网地址说明

代码涉及公网地址参考 public_address_statement.md

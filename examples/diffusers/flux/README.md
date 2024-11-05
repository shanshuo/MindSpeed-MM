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
  commit_id= a98a839de75f1ad82d8d200c3bc2e4ff89929081
  ```

## 微调

### 环境搭建

【模型开发时推荐使用配套的环境版本】

|           软件            | [版本](https://www.hiascend.com/zh/) |
| :-----------------------: |:----------------------------------:|
|          Python           |                3.8                 |
|          Driver           |         AscendHDK 24.1.RC3          |
|         Firmware          |         AscendHDK 24.1.RC3          |
|           CANN            |             CANN 8.0.RC3             |
|           Torch           |            2.1.0            |
|         Torch_npu         |           release v6.0.RC3           |

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
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout a98a839de75f1ad82d8d200c3bc2e4ff89929081
    cp -r ../MindSpeed-MM/examples/diffusers/flux ./examples/dreambooth
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install e .
    vim examples/dreambooth/requirements_flux.txt #修改版本：torchvision==0.16.0, torch==2.1.0, accelerate==0.33.0, 添加deepspeed==0.15.2
    pip install -r examples/dreambooth/requirements_flux.txt # 安装对应依赖
    ```

<a id="jump2"></a>

## 微调

1. 【准备微调数据集】

    - 只包含图片的训练数据集，如非deepspeed脚本使用训练数据集dog:[下载地址](https://huggingface.co/datasets/diffusers/dog-example)，并将dog文件夹转移到`examples/dreambooth/`目录下

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

    - 如用自己的微调数据集，需在shell脚本中将`input_dir`修改为`dataset_name`：

    ```shell
    dataset_name="/path/customized_datasets" # 数据集路径
    ```

    在shell脚本`accelerate launch`目录下（70行左右）将`instance_data_dir=$instance_dir \` 修改为 `dataset_name=$dataset_name`，并将`instance_prompt`与`validation_prompt`改为与自己数据集所匹配的prompt:

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_flux.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --instance_prompt="a prompt that is suitable for your own dataset" \
      --validation_prompt="a validation prompt based on your own dataset" \
    ```

2. 【配置 FLUX 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[FLUX.1-dev模型](https://huggingface.co/black-forest-labs/FLUX.1-dev) `model_name`模型

    ```bash
    export model_name="black-forest-labs/FLUX.1-dev" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    model_name="black-forest-labs/FLUX.1-dev" # 预训练模型路径
    batch_size=16
    max_train_steps=5000
    mixed_precision="bf16" # 混精
    resolution=256
    config_file="pretrain_${mixed_precision}_accelerate_config.yaml"
    ```

3. 【修改代码文件】

    1. 在 `src/diffusers/models/embeddings.py` 文件里，在 `class FluxPosEmbed(nn.Module):` 下的 **第760行左右** 找到代码： `freqs_dtype = torch.float32 if is_mps else torch.float64` 进行修改, 请修改为：`freqs_dtype = torch.float32`

        ```shell
        # 修改为freqs_dtype = torch.float32
        vim src/diffusers/models/embeddings.py
        ```

    2. 打开`train_dreambooth_flux.py`文件并在62行附近添加代码

        ```shell
        cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
        vim train_dreambooth_flux.py # 进入Python文件
        ```

        ```python
        # 添加代码到train_dreambooth_flux.py 62行附近
        from patch_flux import TorchPatcher, config_gc
        TorchPatcher.apply_patch()
        config_gc()
        ```

    3. 【Optional】Ubuntu系统需在1701行附近 添加 `accelerator.print("")`

        ```python
        if global_step >= args.max_train_steps:
          break
        accelerator.print("")
        ```

3. 【启动 FLUX 微调脚本】

    本任务主要提供finetune_flux_dreambooth_deepspeed_bf16.sh脚本，该脚本为FLUX微调脚本，支持多卡训练。

    ```shell
    bash finetune_flux_dreambooth_deepspeed_bf16.sh 
    ```

### 性能

#### 吞吐

FLUX 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Flux-全参微调  |  55.23  |     16      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | Flux-全参微调  |  53.65 |     16      | bf16 | 2.1 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

```shell
cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
```

【FLUX模型推理】

```shell
vim infer_flux_text2img_bf16.py # 进入运行推理的Python文件
```

  1. 修改路径

      ```python
      MODEL_PATH = "/black-forest-labs/FLUX.1-dev"  # FLUX模型路径
      ```

  2. 运行代码

      ```shell
      python infer_flux_text2img_bf16.py
      ```

  【DREAMBOOTH微调FLUX模型推理】

  ```shell
  vim infer_flux_text2img_dreambooth_bf16.py
  ```

  1. 修改路径

      ```python
      MODEL_PATH = "./output_FLUX_dreambooth"  # Dreambooth微调保存模型路径
      ```

  2. 运行代码

      ```shell
      python infer_flux_text2img_dreambooth_bf16.py
      ```

<a id="jump3"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  文生图  | 1.16 | bf16 | 2.1 |
| 竞品A | 8p |  文生图  | 1.82 | bf16 | 2.1 |
| Atlas 900 A2 PODc |8p |  文生图微调  | 1.12 | bf16 | 2.1 |
| 竞品A | 8p |  文生图微调  | 1.82 | bf16 | 2.1 |

## 引用

### 公网地址说明

代码涉及公网地址参考 public_address_statement.md

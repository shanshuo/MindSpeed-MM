# InternVL2 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

<table border="0">
  <tr>
    <th>软件</th>
    <th>版本</th>
    <th>安装指南</th>
  </tr>
  <tr>
    <td> Python </td>
    <td> 3.10 </td>
  </tr>
  <tr>
    <td> Driver </td>
    <td> AscendHDK 24.1.RC3 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Firmware </td>
    <td> AscendHDK 24.1.RC3 </td>
  </tr>
  <tr>
    <td> CANN </td>
    <td> CANN 8.0.RC3 </td>
    <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Torch </td>
    <td> 2.1.0 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td> Torch_npu </td>
    <td> release v6.0.RC3 </td>
  </tr>
</table>

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
    mkdir logs
    mkdir dataset
    mkdir ckpt
```

<a id="jump1.2"></a>

#### 2. 环境搭建

torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)

```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp310-cp310m-linux_aarch64.whl

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout 4c6847e6fda0a458914fd2ea664f6d09a8be300e
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..
    # 替换MindSpeed中的文件
    cp examples/internvl2/dot_product_attention.py MindSpeed/mindspeed/core/transformer/dot_product_attention.py
    
    # 安装其余依赖库
    pip install -e .
```

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B/tree/main)；
- [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)；
- [InternVL2-Llama3-76B](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B/tree/main)；

将模型权重保存在`raw_ckpt/InternVL2-2B`目录。

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeeed-MM修改了部分原始网络的结构名称，使用`examples/internvl2/internvl_convert_to_mm_ckpt.py`脚本对原始预训练权重进行转换。该脚本实现了从huggingface权重到MindSpeed-MM权重的转换以及PP（Pipeline Parallel）权重的切分。

以InternVL2-2B为例，`inernvl_convert_to_mm_ckpt.py`的入参`load_dir`、`save_dir`、`pipeline_layer_index`、`num_layers`如下：

```shell
  --model-size 2B
  --load-dir raw_ckpt/InternVL2-2B # huggingface权重目录
  --save-dir pretrained/InternVL2-2B  # 转换后保存目录
  --trust-remote-code True  # 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
```

启动脚本

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python examples/internvl2/internvl_convert_to_mm_ckpt.py \
    --model-size 2B \
    --load-dir raw_ckpt/InternVL2-2B \
    --save-dir pretrained/InternVL2-2B \
    --trust-remote-code True
```

同步修改`examples/internvl2/finetune_internvl2_2b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重`raw_ckpt/InternVL2-2B`进行区分。

```shell
LOAD_PATH="pretrained/InternVL2-2B"
```

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压[InternVL-Finetune](https://huggingface.co/datasets/OpenGVLab/InternVL-Chat-V1-2-SFT-Data)数据集到`dataset/playground`目录下，以数据集ai2d为例，解压后的数据结构如下：

   ```
   $playground
   ├── data
       ├── ai2d
           ├── abc_images
           ├── images
   ├── opensource
       ├── ai2d_train_12k.jsonl
   ```

---

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`from_pretrained`、`data_path`、`data_folder`等字段。

以InternVL2-2B为例，`data_2B.json`进行以下修改，注意`tokenizer_config`的权重路径为转换前的权重路径。

```json
{
  "dataset_param": {
      ...
      "basic_parameters": {
          "data_path": "dataset/playground/opensource/ai2d_train_12k.jsonl",
          "data_folder": "dataset/playground/data/ai2d"
      },
      ...
      "tokenizer_config": {
          ...
          "from_pretrained": "raw_ckpt/InternVL2-2B",
          ...
      },
      ...
  },
  ...
}
```

【模型保存加载配置】

根据实际情况配置`examples/internvl2/finetune_internvl2_xx.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）, 以InternVL2-2B为例：

```shell
...
# 加载路径
LOAD_PATH="ckpt/InternVL2-2B"
# 保存路径
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # 不加载优化器状态，若需加载请移除
    --no-load-rng \  # 不加载随机数状态，若需加载请移除
    --no-save-optim \  # 不保存优化器状态，若需保存请移除
    --no-save-rng \  # 不保存随机数状态，若需保存请移除
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # 日志间隔
    --save-interval 5000 \  # 保存间隔
    ...
"
```

若需要加载指定迭代次数的权重、优化器等状态，需将加载路径`LOAD_PATH`设置为保存文件夹路径`LOAD_PATH="save_dir"`，并修改`latest_checkpointed_iteration.txt`文件内容为指定迭代次数

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

【单机运行配置】

配置`examples/internvl2/finetune_internvl2_xx.sh`参数如下

```shell
    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以InternVL2-2B为例，启动微调训练任务。

```shell
    bash examples/internvl2/finetune_internvl2_2B.sh
```

<a id="jump5"></a>

## 推理

#### 1. 准备工作
配置脚本前需要完成前置准备工作，包括：环境安装、权重下载及转换，详情可查看对应章节。

#### 2. 配置参数
【参数配置】

修改inference_xx.json文件，包括`image_path`、`prompt`、`from_pretrained`以及tokenizer的`from_pretrained`等字段。
以InternVL2-2B为例，按实际情况修改inference_2B.json，注意tokenizer_config的权重路径为转换前的权重路径。
```json
{   
    "image_path": "./examples/internvl2/view.jpg",
    "prompts": "Please describe the image shortly.",
    "model_id": "InternVLPipeline",
    "from_pretrained": "./ckpt/InternVL2-2B/release/mp_rank_00/model_optim_rng.pt",
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2-2B",
        ...
    },
    ...
}
```
【启动脚本配置】
按实际情况修改inference_internvl.sh脚本，
```shell
    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    ...
    MM_MODEL="./examples/internvl2/inference_2B.json"
```
#### 3. 启动推理

```shell
  bash examples/internvl2/inference_internvl.sh
```

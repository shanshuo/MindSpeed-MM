# CogVideoX 使用指南

<p align="left">
</p>

## 目录
- [CogVideoX 使用指南](#cogvideox-使用指南)
  - [目录](#目录)
  - [支持任务列表](#支持任务列表)
  - [环境安装](#环境安装)
      - [仓库拉取](#仓库拉取)
      - [环境搭建](#环境搭建)
      - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
      - [VAE下载](#vae下载)
      - [transformer文件下载](#transformer文件下载)
      - [T5模型下载](#t5模型下载)
      - [权重转换](#权重转换)
  - [数据集准备及处理](#数据集准备及处理)
  - [预训练](#预训练)
      - [准备工作](#准备工作)
      - [配置参数](#配置参数)
      - [启动预训练](#启动预训练)
  - [推理](#推理)
      - [准备工作](#准备工作-1)
      - [配置参数](#配置参数-1)
      - [启动推理](#启动推理)

---
<a id="jump1"></a>
## 支持任务列表
支持以下模型任务类型

|      模型      | 任务类型 | 任务列表 | 是否支持 |
|:------------:|:----:|:----:|:-----:|
| CogVideoX-5B | t2v  |预训练  | ✔ |
| CogVideoX-5B | t2v  |在线推理 | ✔ |
| CogVideoX-5B | i2v  |预训练  | ✔ |
| CogVideoX-5B | i2v  |在线推理 | ✔ |

<a id="jump2"></a>
## 环境安装

【模型开发时推荐使用配套的环境版本】

|           软件            | [版本](https://www.hiascend.com/hardware/firmware-drivers/commercial?product=4&model=26) |
| :-----------------------: |:--------------------------------------------------------------------------------------:|
|          硬件配置         |                          Atlas 800T A2 <br> Atlas 900 A2 PoD                           |
|          Python           |                                          3.10                                          |
|          Driver           |                                   AscendHDK 24.1.RC3                                   |
|         Firmware          |                                   AscendHDK 24.1.RC3                                   |
|           CANN            |                                      CANN 8.0.RC3                                      |
|           Torch           |                                         2.1.0                                          |
|         Torch_npu         |                                    release v6.0.RC3                                    |

<a id="jump2.1"></a>
#### 仓库拉取

```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
```
<a id="jump2.2"></a>
#### 环境搭建


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

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout 5dc1e83b
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -e .
```
<a id="jump2.3"></a>
#### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---
<a id="jump3"></a>
## 权重下载及转换

<a id="jump3.1"></a>
#### VAE下载

+ [VAE下载链接](https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1)

<a id="jump3.2"></a>
#### transformer文件下载
+ [CogVideoX-5B-t2v](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX-5B-i2v](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)

<a id="jump3.3"></a>
#### T5模型下载
仅需下载tokenizer和text_encoder目录的内容：[下载链接](https://huggingface.co/THUDM/CogVideoX-5b/tree/main)

预训练权重结构如下：
   ```
   CogVideoX-5B
   ├── text_encoder
   │   ├── config.json
   │   ├── model-00001-of-00002.safetensors
   │   ├── model-00002-of-00002.safetensors
   │   └── model.safetensors.index.json   
   ├── tokenizer
   │   ├── added_tokens.json
   │   ├── special_tokens_map.json
   │   ├── spiece.model
   │   └── tokenizer_config.json   
   ├── transformer
   │   ├── 1000 (or 1)
   │   │   └── mp_rank_00_model_states.pt
   │   └── latest
   └── vae
       └── 3d-vae.pt
   ```

<a id="jump3.4"></a>
#### 权重转换
权重转换source_path参数请配置transformer权重文件的路径：
```bash
python examples/cogvideox/cogvideox_convert_to_mm_ckpt.py --source_path <your source path> --target_path <target path> --task t2v --tp_size 1 --mode split
```

---
<a id="jump4"></a>
## 数据集准备及处理

数据集格式应该如下：
```
.
├── data.jsonl
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```
每个 txt 与视频同名，为视频的标签。视频与标签应该一一对应。

data.jsonl文件内容如下示例：
```
{"file": "dataPath/1.mp4", "captions": "Content from 1.txt"}
{...}
...
```

---
<a id="jump5"></a>
## 预训练

<a id="jump5.1"></a>
#### 准备工作
配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump5.2"></a>
#### 配置参数
需根据实际任务情况修改`model_cogvideox.json`、`model_cogvideox_i2v.json`和`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段。

在sh启动脚本中可以修改运行卡数：
```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=29501
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```
<a id="jump5.3"></a>
#### 启动预训练

t2v任务启动预训练
```shell
    bash examples/cogvideox/pretrain_cogvideox_t2v.sh
```
i2v任务启动预训练
```shell
    bash examples/cogvideox/pretrain_cogvideox_i2v.sh
```
---

<a id="jump6"></a>
## 推理

<a id="jump6.1"></a>
#### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

<a id="jump6.2"></a>
#### 配置参数

检查如下配置是否完成

| 配置文件 |               修改字段               |                修改说明                 |
|------|:--------------------------------:|:-----------------------------------:|
|  examples/cogvideox/inference_model.json    |         from_pretrained          |            修改为下载的权重所对应路径            |
|   examples/cogvideox/samples_prompts.txt   |               文件内容               |      可自定义自己的prompt，一行为一个prompt      |

如果使用训练后保存的权重进行推理，需要使用脚本进行转换，权重转换source_path参数请配置训练时的保存路径
```bash
python examples/cogvideox/cogvideox_convert_to_mm_ckpt.py --source_path <your source path> --target_path <target path> --task t2v --tp_size 1 --mode merge
```

<a id="jump6.3"></a>
#### 启动推理

```bash
bash examples/cogvideox/inference_cogvideox.sh
```

---

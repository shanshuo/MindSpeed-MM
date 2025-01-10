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
# 建议从原仓编译安装 

# 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.6.0
git checkout 6891dfa1a6a12460fdc49d54c6ee43e0f967b4ae
pip install -r requirements.txt 
pip install -e .
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
+ [CogVideoX1.0-5B-t2v](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX1.0-5B-i2v](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)
+ [CogVideoX1.5-5B-i2v&t2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main)

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
python examples/cogvideox/cogvideox_convert_to_mm_ckpt.py --source_path <your source path> --target_path <target path> --task t2v --tp_size 1 --pp_size 10 11 11 10 --num_layers 42 --mode split
```
当开启PP时，--pp_size 后参数值个数与PP的数值相等，并且参数之和与--num_layers 参数相等，举例：当PP=4, --num_layers 4, --pp_size 1 1 1 1; 当PP=4, --num_layers 42, --pp_size 10 11 11 10

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
需根据实际任务情况修改`pretrain_cogvideox_i2v.sh`/`pretrain_cogvideox_i2v_1.5.sh`、`pretrain_cogvideox_t2v.sh`/`pretrain_cogvideox_t2v_1.5.sh`和`data.json`中的权重和数据集路径，包括`LOAD_PATH`、`SAVE_PATH`、`data_path`、`data_folder`字段。`LOAD_PATH`字段中填写的权重路径位置一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。

根据实际情况修改`model_cogvideox_t2v.json`/`model_cogvideox_t2v_1.5.json`、`model_cogvideox_i2v.json`/`model_cogvideox_i2v_1.5.json`、`data.json`文件中VAE及T5模型文件的实际路径。


`model_cogvideox_t2v.json`/`model_cogvideox_i2v.json`文件中的`head_dim`字段原模型默认配置为64。此字段调整为128会更加亲和昇腾。

在sh启动脚本中可以修改运行卡数(NNODES为节点数，GPUS_PER_NODE为每个节点的卡数，相乘即为总运行卡数)：
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

t2v 1.0版本任务启动预训练
```shell
bash examples/cogvideox/t2v_1.0/pretrain_cogvideox_t2v.sh
```
t2v 1.5版本任务启动预训练
```shell
bash examples/cogvideox/t2v_1.5/pretrain_cogvideox_t2v_1.5.sh
```
i2v 1.0版本任务启动预训练
```shell
bash examples/cogvideox/i2v_1.0/pretrain_cogvideox_i2v.sh
```
i2v 1.5版本任务启动预训练
```shell
bash examples/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh
```
---

<a id="jump6"></a>
## 推理

<a id="jump6.1"></a>
#### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

<a id="jump6.2"></a>
#### 配置参数

检查对应配置是否完成

| t2v配置文件                                           |               修改字段               |                修改说明                 |
|---------------------------------------------------|:--------------------------------:|:-----------------------------------:|
| examples/cogvideox/t2v_*/inference_model_t2v.json |         from_pretrained          |            修改为下载的权重所对应路径            |
| examples/cogvideox/samples_prompts.txt            |               文件内容               |      可自定义自己的prompt，一行为一个prompt      |


| i2v配置文件                                           |               修改字段               |       修改说明       |
|---------------------------------------------------|:--------------------------------:|:----------------:|
| examples/cogvideox/i2v_*/inference_model_i2v.json |         from_pretrained          |  修改为下载的权重所对应路径   |
| examples/cogvideox/samples_i2v_images.txt         |               文件内容               |       图片路径       |
| examples/cogvideox/samples_i2v_prompts.txt        |               文件内容               |    自定义prompt     |


如果使用训练后保存的权重进行推理，需要使用脚本进行转换，权重转换source_path参数请配置训练时的保存路径
```bash
python examples/cogvideox/cogvideox_convert_to_mm_ckpt.py --source_path <your source path> --target_path <target path> --task t2v --tp_size 1 --mode merge
```

<a id="jump6.3"></a>
#### 启动推理
t2v 启动推理脚本

```bash
bash examples/cogvideox/t2v_1.0/inference_cogvideox_t2v.sh
```
i2v 启动推理脚本

```bash
bash examples/cogvideox/i2v_1.0/inference_cogvideox_i2v.sh
```

---
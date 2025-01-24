# HunyuanVideo使用指南

- [HunyuanVideo使用指南](#hunyuanvideo使用指南)
  - [环境安装](#环境安装)
    - [仓库拉取](#仓库拉取)
    - [环境搭建](#环境搭建)
    - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
    - [TextEncoder下载](#textencoder下载)
    - [HunyuanVideoDiT与VAE下载](#hunyuanvideodit与vae下载)
    - [权重转换](#权重转换)
  - [预训练](#预训练)
  - [推理](#推理)
    - [准备工作](#准备工作)
    - [参数配置](#参数配置)
    - [启动推理](#启动推理)


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
    <td> AscendHDK 24.1.0 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Firmware </td>
    <td> AscendHDK 24.1.0 </td>
  </tr>
  <tr>
    <td> CANN </td>
    <td> CANN 8.0.0 </td>
    <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/800/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Torch </td>
    <td> 2.1.0 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/600/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td> Torch_npu </td>
    <td> release v6.0.0 </td>
  </tr>
</table>

### 仓库拉取

```shell
git clone https://gitee.com/ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.6.0
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```

### 环境搭建


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
git checkout b35b09e20da8651aee0d742748591c0a9259270e
pip install -r requirements.txt 
pip install -e .
cd ..

# 安装其余依赖库
pip install -e .
```

### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

## 权重下载及转换

### TextEncoder下载
+ [llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers)
+ [clip-vit-large](https://huggingface.co/openai/clip-vit-large-patch14)

### HunyuanVideoDiT与VAE下载
+ [tencent/HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)

下载后的权重结构如下
```shell
HunyuanVideo
  ├──README.md
  ├──hunyuan-video-t2v-720p
  │  ├──transformers
  │  │  ├──mp_rank_00_model_states.pt
  │  ├──vae
  │  │  ├──config.json
  │  │  ├──pytorch_model.pt
```
其中`HunyuanVideo/hunyuan-video-t2v-720p/transformers`是transformer部分的权重，`HunyuanVideo/hunyuan-video-t2v-720p/vae`是VAE部分的权重

### 权重转换
需要对`llava-llama3-8b`模型进行权重转换，运行权重转换脚本：
```shell
python examples/convert_ckpt_to_mm.py --module text_encoder --source_path <llava-llama-3-8b> --target_path <llava-llama-3-8b-text-encoder-tokenizer>
```

需要对hunyuanvideo的transformer部分进行权重转换，运行权重转换脚本：
```shell
python examples/convert_ckpt_to_mm.py --source_path <hunyuan-video-t2v-720p/transformers/mp_rank_00/model_states.pt> --target_path <./ckpt/hunyuanvideo>
```

权重转换脚本的参数说明如下：
|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --module |  转换text encoder部分或transformer部分 | "dit" （转换transformer权重） |
| --source_path | 原始权重路径 | ./transformers/mp_rank_00/model_states.pt |
| --target_path | 转换后的权重保存路径 | ./ckpt/hunyuanvideo |

---

## 预训练
Comming soon

## 推理

### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

### 参数配置

检查模型权重路径、并行参数等配置是否完成

| 配置文件                                           |               修改字段               |                修改说明                 |
|---------------------------------------------------|:--------------------------------:|:-----------------------------------|
| examples/hunyuanvideo/inference_model.json |         from_pretrained          |            修改为下载的权重所对应路径（包括VAE、Text Encoder）            |
| examples/hunyuanvideo/samples_prompts.txt            |               文件内容               |      可自定义自己的prompt，一行为一个prompt      |
| examples/hunyuanvideo/inference_model.json    |  input_size |  生成视频的分辨率，格式为 [t, h, w] |
| examples/hunyuanvideo/inference_model.json    |  save_path |  生成视频的保存路径 |
| examples/hunyuanvideo/inference_hunyuanvideo.sh   |   LOAD_PATH | 转换之后的transform部分权重路径 |



### 启动推理

```shell
bash examples/hunyuanvideo/inference_hunyuanvideo.sh
```
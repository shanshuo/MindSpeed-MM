# Qwen2_VL 使用指南

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
- [注意事项](#jump6)
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
    mkdir data
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
    # 下载路径参考 https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html
    pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp310-cp310m-linux_aarch64.whl

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout ab39de78be23e88e2c8b0d25edf6135940990c02
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..
    # 替换MindSpeed中的文件
    cp examples/qwen2vl/dot_product_attention.py MindSpeed/mindspeed/core/transformer/dot_product_attention.py
    
    # 安装其余依赖库
    pip install -e .
```

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface库下载对应的模型权重:
- 模型地址: [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main)；

- 模型地址: [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main)；

 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2-VL-*B-Instruct`目录下。(*表示对应的尺寸)
<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用examples/qwen2vl/qwen2vl_convert_to_mm_ckpt.py脚本对原始预训练权重进行转换。该脚本实现了从huggingface权重到MindSpeed-MM权重的转换以及PP（Pipeline Parallel）权重的切分 (目前只支持7B和2B特定的切分方式)。

以Qwen2VL-7B为例
修改qwen2vl_convert_to_mm_ckpt.py中的如下内容,与实际保持一致：

```
hg_ckpt_dir = 'ckpt/hf_path/Qwen2-VL-7B-Instruct'  # huggingface权重目录
mm_save_dir = 'ckpt/mm_path/Qwen2-VL-7B-Instruct'  # 转换后保存目录
vit_hidden_size = 1280  # vit的隐藏层size
vit_attention_heads_num = 16  # vit的注意力heads数
pp_size = 4  # 切分的PPstage数量
vit_num_layers = 32  # vit的总层数
vit_pipeline_num_layers = [32, 0, 0, 0]  # vit在每个卡上切分的层数，和为 vit_num_layers
llm_num_layers = 28  #LLM的总层数
llm_pipeline_num_layers = [1, 6, 11, 10]  # LLM在每个卡上切分的层数，和为 llm_num_layers
```
以Qwen2VL-2B为例
修改qwen2vl_convert_to_mm_ckpt.py中的如下内容,与实际保持一致：

```
hg_ckpt_dir = 'ckpt/hf_path/Qwen2-VL-2B-Instruct'  # huggingface权重目录
mm_save_dir = 'ckpt/mm_path/Qwen2-VL-2B-Instruct'  # 转换后保存目录
vit_hidden_size = 1280  # vit的隐藏层size
vit_attention_heads_num = 16  # vit的注意力heads数
pp_size = 1  # 2B不需要切分PP
vit_num_layers = 32  # vit的总层数
vit_pipeline_num_layers = [32]  # vit在每个卡上切分的层数，和为vit_num_layers
llm_num_layers = 28  #LLM的总层数
llm_pipeline_num_layers = [28]  # LLM在每个卡上切分的层数，和为llm_num_layers
```
  
启动脚本

  ```
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python examples/qwen2vl/qwen2vl_convert_to_mm_ckpt.py
  ```

如果需要用转换后模型训练的话，同步修改examples/qwen2vl/finetune_qwen2vl_7b.sh中的LOAD_PATH参数，该路径为转换后或者切分后的权重，注意与原始权重 hf_path/Qwen2-VL-7B-Instruct进行区分。

```
LOAD_PATH="ckpt/Qwen2-VL-7B-Instruct"
```

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载(以coco2017数据集为例)

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下;

(3)在./data路径下新建文件mllm_format_llava_instruct_data.json，运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py;

   ```
   $playground
   ├── data
       ├── COCO2017
           ├── train2017
           
       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

---
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为`data.json`中
dataset_param->basic_parameters->dataset
从"./data/mllm_format_llava_instruct_data.json"修改为"./data/mllm_format_llava_instruct_data.json,./data/mllm_format_llava_instruct_data2.json"
  
同时注意`data.json`中`dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。如果正式训练时，可以把该参数去掉则读取全部的数据。
<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2VL-7B为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2-VL-7B-Instruct",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
    },
    ...
    }
}
```

【模型保存加载配置】

根据实际情况配置`examples/qwen2vl/finetune_qwen2vl_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/Qwen2-VL-7B-Instruct"
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
(此功能coming soon)

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

【单机运行配置】

配置`examples/qwen2vl/finetune_qwen2vl_7b.sh`参数如下

```shell
    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=29501
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以Qwen2VL-7B为例，启动微调训练任务。

```shell
    bash examples/qwen2vl/finetune_qwen2vl_7b.sh
```

<a id="jump5"></a>

## 推理
#### 1、准备工作（以微调环境为基础，包括环境安装、权重下载及转换-目前支持PP切分的推理）
追加安装：
```
pip install qwen_vl_utils
```
注：如果使用huggingface下载的原始权重，需要权重转换，权重转换步骤中，根据具体需求设置PP切分的参数。

注：如果使用的MindSpeed-MM中保存的权重则无需进行转换，可直接加载(需要保证与训练的切分一致)。
#### 2、配置参数
根据实际情况修改examples/qwen2vl/inference_qwen2vl_7b.json和examples/qwen2vl/inference_qwen2vl_7b.sh中的路径配置，包括tokenizer的加载路径tokenizer_name_or_path、以及图片处理器的路径image_processer_path。需注意

1、tokenizer_name_or_path配置的路径为从huggingface下载的原始Qwen2-VL-7B-Instruct路径。

2、shell文件中的LOAD_PATH的路径为经过权重转换后的模型路径(可PP切分)。

#### 3、启动推理
```shell
    bash examples/qwen2vl/inference_qwen2vl_7b.sh
```
注：单卡推理需打开FA，否则可能会显存不足报错，开关--use-flash-attn 默认已开，确保FA步骤完成即可。如果使用多卡推理则需要调整相应的PP参数和NPU使用数量的NPUS_PER_NODE参数。以PP4为例，shell修改参数如下：

```shell
NPUS_PER_NODE=4 # 可用几张卡 要大于 PP*TP*CP
PP=4 #PP并行参数
```

<a id="jump6"></a>

## 注意事项：
1. 在使用流水线并行策略进行多机训练可能会出现卡住现象，可参考[此处](https://gitee.com/ascend/MindSpeed/pulls/1627/files)修改。
2. 模型相关参数在examples/qwen2vl/model_xb.json 修改，训练相关参数在 finetune_xx.sh修改。




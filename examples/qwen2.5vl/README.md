# Qwen2_5_VL 使用指南

<p align="left">
</p>

[toc]

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

#### 1. 仓库拉取

```shell
git clone https://gitee.com/ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
mkdir logs
mkdir data
mkdir ckpt
```

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
# 建议从原仓编译安装

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.8.0
git checkout 3f09d6736571cf1e30f8ac97de77982d0ab32cc5
pip install -r requirements.txt
pip3 install -e .
cd ..
# 替换MindSpeed中的文件
cp examples/qwen2vl/dot_product_attention.py MindSpeed/mindspeed/core/transformer/dot_product_attention.py

# 安装其余依赖库
pip install -e .

# 安装transformers指定版本
git clone git@github.com:huggingface/transformers.git
cd transformers
git checkout fa56dcc2a
pip install -e .

```

## 权重下载及转换

#### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)；


 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2.5-VL-7B-Instruct`目录下。
<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分。参考[权重转换工具](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)

```bash
# 7b
mm-convert  Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1

# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```

如果需要用转换后模型训练的话，同步修改`examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```


## 数据集准备及处理

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

#### 2.纯文本或有图无图混合训练数据(以LLaVA-Instruct-150K为例)

现在本框架已经支持纯文本/混合数据（有图像和无图像数据混合训练）。

在数据构造时，对于包含图片的数据，需要保留`image`这个键值。

```python
{
  "id": your_id,
  "image": your_image_path,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

在数据构造时，对于纯文本数据，可以去除`image`这个键值。

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

## 微调

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2.5VL-7B为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-7B-Instruct",
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

【模型保存加载及日志信息配置】

根据实际情况配置`examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
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
    --log-tps \  # 增加此参数可使能在训练中打印每步语言模块的平均序列长度，并在训练结束后计算每秒吞吐tokens量。
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

配置`examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`参数如下

```shell
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
NPUS_PER_NODE=8
MASTER_ADDR=locahost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```
注意，当开启PP时，`model.json`中配置的`vision_encoder`和`text_decoder`的`pipeline_num_layer`参数控制了各自的PP切分策略。对于流水线并行，要先处理`vision_encoder`再处理`text_decoder`。
比如7b默认的值`[32,0,0,0]`、`[1,10,10,7]`，其含义为PP域内第一张卡先放32层`vision_encoder`再放1层`text_decoder`、第二张卡放`text_decoder`接着的10层、第三张卡放`text_decoder`接着的10层、第四张卡放`text_decoder`接着的7层，`vision_encoder`没有放完时不能先放`text_decoder`（比如`[30,2,0,0]`、`[1,10,10,7]`的配置是错的）

同时注意，如果某张卡上的参数全部冻结时会导致没有梯度（比如`vision_encoder`冻结时PP配置`[30,2,0,0]`、`[0,11,10,7]`），需要在`finetune_qwen2_5_vl_7b.sh`中`GPT_ARGS`参数中增加`--enable-dummy-optimizer`，参考[dummy_optimizer特性文档](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/dummy_optimizer.md)。

#### 3. 启动微调

以Qwen2.5VL-7B为例，启动微调训练任务。

```shell
bash examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
```

## 环境变量声明
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化  
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景  
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核  
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s  
NPU_ASD_ENABLE： 控制是否开启Ascend Extension for PyTorch的特征值检测功能，未设置或0：关闭特征值检测，1：表示开启特征值检测，只打印异常日志，不告警，2：开启特征值检测，并告警，3：开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据  
ASCEND_LAUNCH_BLOCKING： 控制算子执行时是否启动同步模式，0：采用异步方式执行，1：强制算子采用同步模式运行  
ACLNN_CACHE_LIMIT： 配置单算子执行API在Host侧缓存的算子信息条目个数  
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量
# Qwen2_VL 使用指南

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
```

## 权重下载及转换

#### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main)；

- 模型地址: [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main)；

- 模型地址: [Qwen2-VL-72B](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct/tree/main)；

 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2-VL-*B-Instruct`目录下。(*表示对应的尺寸)
<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。该工具实现了huggingface权重和MindSpeed-MM权重的互相转换以及PP（Pipeline Parallel）权重的重切分。参考[权重转换工具](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/权重转换工具.md)

```bash
# 2b
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-2B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-2B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [28] \
  --cfg.parallel_config.vit_pp_layers [32] \
  --cfg.parallel_config.tp_size 1

# 7b
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1

# 72b
mm-convert  Qwen2VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-72B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-72B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [5,11,11,11,11,11,11,9] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0,0,0,0,0] \
  --cfg.parallel_config.tp_size 2
# 其中：
# mm_dir: 转换后保存目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```

如果需要用转换后模型训练的话，同步修改`examples/qwen2vl/finetune_qwen2vl_7b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2-VL-7B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2-VL-7B-Instruct"
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

以Qwen2VL-7B为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

**注意`cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突**。

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

【模型保存加载及日志信息配置】

根据实际情况配置`examples/qwen2vl/finetune_qwen2vl_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/mm_path/Qwen2-VL-7B-Instruct"
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

配置`examples/qwen2vl/finetune_qwen2vl_7b.sh`参数如下

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

同时注意，如果某张卡上的参数全部冻结时会导致没有梯度（比如`vision_encoder`冻结时PP配置`[30,2,0,0]`、`[0,11,10,7]`），需要在`finetune_qwen2vl_7b.sh`中`GPT_ARGS`参数中增加`--enable-dummy-optimizer`，参考[dummy_optimizer特性文档](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/dummy_optimizer.md)。

#### 3. 启动微调

以Qwen2VL-7B为例，启动微调训练任务。

```shell
bash examples/qwen2vl/finetune_qwen2vl_7b.sh
```

## 推理

#### 1、准备工作（以微调环境为基础，包括环境安装、权重下载及转换-目前支持PP切分的推理）

追加安装：

```shell
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

## 训练后权重转回huggingface格式

MindSpeed-MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回huggingface格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

```bash
mm-convert  Qwen2VLConverter mm_to_hf \
  --cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2-VL-7B-Instruct" \
  --cfg.mm_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.parallel_config.tp_size 1
# 其中：
# save_hf_dir: mm微调后转换回hf模型格式的目录
# mm_dir: 微调后保存的权重目录
# hf_dir: huggingface权重目录
# llm_pp_layers: llm在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# vit_pp_layers: vit在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致
# tp_size: tp并行数量，注意要和微调启动脚本中的配置一致
```


## 训练后重新切分权重

权重下载及转换部分会把权重进行pp切分和tp切分，在微调后，如果需要对权重重新进行切分，可使用`mm-convert`权重转换工具对微调后的权重进行切分。


```bash
mm-convert  Qwen2VLConverter resplit \
  --cfg.source_dir "ckpt/mm_path/Qwen2-VL-7B-Instruct" \
  --cfg.target_dir "ckpt/mm_resplit_pp/Qwen2-VL-7B-Instruct" \
  --cfg.source_parallel_config.llm_pp_layers [1,10,10,7] \
  --cfg.source_parallel_config.vit_pp_layers [32,0,0,0] \
  --cfg.source_parallel_config.tp_size 1 \
  --cfg.target_parallel_config.llm_pp_layers [4,24] \
  --cfg.target_parallel_config.vit_pp_layers [32,0] \
  --cfg.target_parallel_config.tp_size 1
# 其中
# source_dir: 微调后保存的权重目录
# target_dir: 希望重新pp切分后保存的目录
# source_parallel_config.llm_pp_layers: 微调时llm的pp配置
# source_parallel_config.vit_pp_layers: 微调时vit的pp配置
# source_parallel_config.tp_size: 微调时tp并行配置
# target_parallel_config.llm_pp_layers: 期望的重切分llm模块切分层数
# target_parallel_config.vit_pp_layers: 期望的重切分vit模块切分层数
# target_parallel_config.tp_size: 期望的tp并行配置（tp_size不能超过原仓config.json中的num_key_value_heads）
```


## Qwen2vl支持视频理解

### 1、加载视频数据集

数据集中的视频数据集取自llamafactory，https://github.com/hiyouga/LLaMA-Factory/tree/main/data

视频取自mllm_demo_data，使用时需要将该数据放到自己的data文件夹中去，同时将llamafactory上的mllm_video_demo.json也放到自己的data文件中

以data_72b.json为例加载数据集：参照data_72b_video.json


### 2、修改模型配置

以72b为例，需要修改model_72b.json：
```
"img_context_token_id": 151656
```
## Qwen2vl支持视频推理
配置修改
以7b模型推理为例，修改inference_qwen2vl_7b.json
```
"img_context_token_id": 151656
```
修改prompts内容中添加对视频的描述
```
"prompts": "Describe this video and keep it within 100 words."
```
支持视频的推理将image_path修改为video_path，原来加载的图片的路径改为视频路径
视频数据样例：
https://github.com/hiyouga/LLaMA-Factory/blob/main/data/mllm_demo_data/1.mp4

暂不支持image_path与video_path同时存在，不支持img和video混合推理

## 评测

### 数据集准备

当前模型支持AI2D(test)、ChartQA(test)、Docvqa(val)、MMMU(val)四种数据集的评测。
数据集参考下载链接：

- [MMMU_DEV_VAL](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)
- [DocVQA_VAL](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)
- [AI2D_TEST](https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv)
- [ChartQA_TEST](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv)

### 参数配置

如果要进行评测需要将要评测的数据集名称和路径传到examples/qwen2vl/evaluate_qwen2vl_7b.json
需要更改的字段有

- `tokenizer`中的`from_pretrained`为huggingface的Qwen2-VL的权重，参考readme上面链接自行下载传入
- `dataset_path`为上述评测数据集的本地路径
- `evaluation_dataset`为评测数据集的名称可选的名称有(`ai2d_test`、`mmmu_dev_val`、`docvqa_val`、`chartqa_test`)， **注意**：需要与上面的数据集路径相对应。
- `result_output_path`为评测结果的输出路径，**注意**：每次评测前需要将之前保存在该路径下评测文件删除。

```json
    "tokenizer": {
        "from_pretrained": "./Qwen2-VL-7B-Instruct",

    },
    "dataset_path": "./AI2D_TEST.tsv",
    "evaluation_dataset":"ai2d_test",
    "evaluation_model":"qwen2_vl_7b",
    "result_output_path":"./evaluation_outputs/"

```

examples/qwen2vl/evaluate_qwen2vl_7b.json改完后，需要将json文件的路径传入到examples/qwen2vl/evaluate_qwen2vl_7b.sh MM_MODEL字段中。

以及需要将上面提到的权重转换后模型传入examples/qwen2vl/evaluate_qwen2vl_7b.sh中的LOAD_PATH字段中。

```shell
MM_MODEL=examples/qwen2vl/evaluate_qwen2vl_7b.json
LOAD_PATH="./qwen_7b_pp1/Qwen2-VL-7B-Instruct"

```
评测支持多卡DP评测需要更改的配置,为NPU卡数量

```shell
NPUS_PER_NODE=1
```

### 启动评测
评测额外依赖一些python包，使用下面命令进行安装

```shell
pip install -e ".[evaluate]"
```

启动shell开始评测
```shell
bash examples/qwen2vl/evaluate_qwen2vl_7b.sh
```

评测结果会输出到`result_output_path`路径中，会输出结果文件：

- *.xlsx文件，这个文件会输出每道题的预测结果和答案等详细信息。
- *.csv文件，这个文件会输出统计准确率等数据。

## 注意事项

1. 在 `finetune_xx.sh`里，与模型结构相关的参数并不生效，以`examples/qwen2vl/model_xb.json`里同名参数配置为准，非模型结构的训练相关参数在 `finetune_xx.sh`修改。
2. LoRA为框架通用能力，当前功能已支持，可参考[LoRA特性文档](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/lora_finetune.md)。

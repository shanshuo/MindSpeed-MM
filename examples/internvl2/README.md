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
- [评测](#jump6)
  - [数据集准备](#jump6.1)
  - [配置参数](#jump6.2)
  - [启动评测](#jump6.3)

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
    # 建议从原仓编译安装

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout ab39de78be23e88e2c8b0d25edf6135940990c02
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

将模型权重保存在`raw_ckpt`目录下，例如`raw_ckpt/InternVL2-8B`。

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用`examples/internvl2/internvl2_convert_to_mm_ckpt.py`脚本对原始预训练权重进行转换。该脚本实现了从huggingface权重到MindSpeed-MM权重的转换以及PP（Pipeline Parallel）权重的切分。

以InternVL2-8B为例，`internvl2_convert_to_mm_ckpt.py`的入参`model-size`、`load-dir`、`save-dir`、`trust-remote-code`等如下：

启动脚本

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python examples/internvl2/internvl2_convert_to_mm_ckpt.py \
    --model-size 8B \
    --load-dir raw_ckpt/InternVL2-8B \    # huggingface权重目录
    --save-dir pretrained/InternVL2-8B \  # 转换后的权重保存目录
    --trust-remote-code True    # 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
```

注：8B/26B/76B默认开启Pipeline并行，如需修改并行配置可在脚本的get_model_config函数中修改对应配置。

同步修改`examples/internvl2/finetune_internvl2_8b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重`raw_ckpt/InternVL2-8B`进行区分。

```shell
LOAD_PATH="pretrained/InternVL2-8B"
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

以InternVL2-8B为例，`data_8B.json`进行以下修改，注意`tokenizer_config`的权重路径为转换前的权重路径。

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
          "from_pretrained": "raw_ckpt/InternVL2-8B",
          ...
      },
      ...
  },
  ...
}
```

【模型保存加载配置】

根据实际情况配置`examples/internvl2/finetune_internvl2_xx.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）, 以InternVL2-8B为例：

```shell
...
# 加载路径
LOAD_PATH="ckpt/InternVL2-8B"
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
    NPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以InternVL2-8B为例，启动微调训练任务。

```shell
    bash examples/internvl2/finetune_internvl2_8B.sh
```

<a id="jump5"></a>

## 推理

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：环境安装、权重下载及转换，详情可查看对应章节。（当前仅支持2B和8B推理功能）

推理权重转换命令如下：

```shell
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python examples/internvl2/internvl2_convert_to_mm_ckpt.py \
    --model-size 8B \
    --load-dir raw_ckpt/InternVL2-8B \    # huggingface权重目录
    --save-dir pretrained/InternVL2-8B \  # 转换后的权重保存目录
    --trust-remote-code True \    # 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
    --is-inference           # 推理模式下，pp size会被设置为1
```

#### 2. 配置参数

【参数配置】

修改inference_xx.json文件，包括`image_path`、`prompt`、`from_pretrained`以及tokenizer的`from_pretrained`等字段。
以InternVL2-8B为例，按实际情况修改inference_8B.json，注意tokenizer_config的权重路径为转换前的权重路径。

```json
{
    "image_path": "./examples/internvl2/view.jpg",    # 按实际情况输入图片
    "prompts": "Please describe the image shortly.", # 按实际情况输入提示词
    "model_id": "InternVLPipeline",
    "from_pretrained": "./pretrained/InternVL2-8B/release/mp_rank_00/model_optim_rng.pt", # 注意路径要到.pt文件
    ...
    "tokenizer":{
        ...
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "raw_ckpt/InternVL2-8B",
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
    MM_MODEL="./examples/internvl2/inference_8B.json"
```

#### 3. 启动推理

```shell
  bash examples/internvl2/inference_internvl.sh
```

<a id="jump6"></a>

## 评测

<a id="jump6.1"></a>

### 数据集准备

当前模型支持AI2D(test)、ChartQA(test)、Docvqa(val)、MMMU(val)四种数据集的评测。
数据集参考下载链接：

- [MMMU_DEV_VAL](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)
- [DocVQA_VAL](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)
- [AI2D_TEST](https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv)
- [ChartQA_TEST](https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv)
<a id="jump6.2"></a>

### 参数配置

如果要进行评测需要将要评测的数据集名称和路径传到examples/internvl2/evaluate_internvl2_8B.json
需要更改的字段有

- `from_pretrained` 需要改为模型的权重文件的路径，如果使用的是huggingface的权重则需要进行权重转换(参考前面的权重转换的章节)，如果使用MindSpeed-MM训练出的则不需要进行权重转换。
- `dataset_path` 需要填入上面下载的数据集文件路径。
- `evaluation_dataset` 为评测数据集的名称可选的名称有(`ai2d_test`、`mmmu_dev_val`、`docvqa_val`、`chartqa_test`)， **注意**：需要与上面的数据集路径相对应。
- `result_output_path` 为评测结果的输出路径，**注意**：每次评测前需要将之前保存在该路径下评测文件删除。
- `tokenizer`下面的`from_pretrained`为huggingface下载的InternVL2-8B权重路径。

```json
    "model_id": "InternVLPipeline",
    "from_pretrained": "./internvl8b_mm/release/mp_rank_00/model_optim_rng.pt",
    "dataset_path": "./AI2D_TEST.tsv",
    "evaluation_dataset":"ai2d_test",
    "evaluation_model":"internvl2_8b",
    "result_output_path":"./evaluation_outputs/",

    "tokenizer":{
        "hub_backend": "hf",
        "autotokenizer_name": "AutoTokenizer",
        "from_pretrained": "./InternVL2-8B",
        "model_max_length": 4096,
        "add_eos_token": false,
        "trust_remote_code": true,
        "use_fast": false
    }

```

examples/internvl2/evaluate_internvl2_8B.json改完后，需要将json文件的路径传入到examples/internvl2/evaluate_internvl2_8B.sh MM_MODEL字段中

```shell
MM_MODEL=examples/internvl2/evaluate_internvl2_8B.json
```

评测支持多卡DP推理需要更改的配置,为NPU卡数量

```shell
NPUS_PER_NODE=1
```

<a id="jump6.3"></a>

### 启动评测

启动shell开始推理

```shell
bash examples/internvl2/evaluate_internvl2_8B.sh
```

评测结果会输出到`result_output_path`路径中，会输出结果文件：

- *.xlsx文件，这个文件会输出每道题的预测结果和答案等详细信息。
- *.csv文件，这个文件会输出统计准确率等数据。

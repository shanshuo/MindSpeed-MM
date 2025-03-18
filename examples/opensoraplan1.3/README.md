# OpenSoraPlan1.3.1 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
  - [Decord安装](#jump1.3)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换](#jump2.2)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
  - [数据集处理](#jump3.2)
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
    <td> 3.8 </td>
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

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.8.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
```

<a id="jump1.2"></a>

#### 2. 环境搭建

torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)

```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    # 建议从原仓编译安装

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 59b4e983b7dc1f537f8c6b97a57e54f0316fafb0
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -e .
```

<a id="jump1.3"></a>

#### 3. Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [LanguageBind/Open-Sora-Plan-v1.3.1](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main)：WFVAE模型和SparseVideoDiT模型；

- [DeepFloyd/mt5-xxl](https://huggingface.co/google/mt5-xxl/)： MT5模型；

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，因此需要使用`convert_ckpt_to_mm.py`脚本进行转换，该脚本实现了从hugging face下载的预训练权重到到MindSpeed-MM权重的转换以及TP（Tensor Parallel）权重的切分。

首先修改 examples/opensoraplan1.3/convert_ckpt_to_mm.py 参数

    TP_SIZE = 1  # TP（Tensor Parallel）size，需要和训练脚本的TP保持一致
    PP_SIZE = [] # PP (Pipeline Parallel) size, 需要和训练脚本保持一致
    dit_hg_weight_path = "raw_ckpt/open-sora-plan/any93x640x640/" #huggingface下载的dit预训练权重路径
    dit_mm_save_dir = "mm_ckpt/open-sora-plan/pretrained-checkpoint-dit" #转换到MindSpeed-MM的dit权重存放路径

    vae_hg_weight_path = "raw_ckpt/vae/wfvae.ckpt"  #huggingface下载的vae预训练权重路径
    vae_mm_save_dir = "mm_ckpt/open-sora-plan/pretrained-checkpoint-wfvae" #转换到MindSpeed-MM的vae权重存放路径
---

启动脚本

    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python examples/opensoraplan1.3/convert_ckpt_to_mm.py
---

同步修改examples/opensoraplan1.3/t2v/pretrain_t2v.sh中的--load参数，该路径为转换后或者切分后的权重，注意--load配置的是转换到MindSpeed-MM后的dit权重路径，vae权重路径在pretrain_t2v_model.json中配置

    LOAD_PATH="mm_ckpt/open-sora-plan/pretrained-checkpoint-dit"

---

#### 3. DistTrain模型分离部署权重转换

提供了MM CKPT与DistTrain CKPT之间的权重转换工具。

MM CKPT转DistTrain CKPT：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python examples/opensoraplan1.3/opensoraplan1_3_mm_convert_to_dt_ckpt.py \
  --load-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit \
  --save-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit-dist-train \
  --target-vae-tp-size 1 \
  --target-vae-pp-size 1 \
  --target-vae-cp-size 1 \
  --target-dit-tp-size 1 \
  --target-dit-pp-size 3 \
  --target-dit-cp-size 1 \
  --target-dit-pp-layers '[10,11,11]'
```

DistTrain CKPT转MM CKPT：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python examples/opensoraplan1.3/opensoraplan1_3_dt_convert_to_mm_ckpt.py \
  --load-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit-dist-train \
  --save-dir mm_ckpt/open-sora-plan/pretrained-checkpoint-dit-dist-train-to-mm \
  --target-tp-size 1 \
  --target-pp-size 4 \
  --target-cp-size 1 \
  --target-dit-pp-layers '[8,8,8,8]'
```

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压[pixabay_v2](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/pixabay_v2_tar)数据集和对应[标注文件](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/anno_json)，获取数据结构如下：

   ```
   $pixabay_v2
   ├── v1.1.0_HQ_part3.json
   ├── folder_01
   ├── ├── video0.mp4
   ├── ├── video1.mp4
   ├── ├── ...
   ├── folder_02
   ├── folder_03
   └── ...
   ```

---
<a id="jump3.2"></a>

#### 2. 数据集处理

根据实际下载的数据，过滤标注文件，删去标注的json文件中未下载的部分；
修改data.txt中的路径，示例如下:

   ```
/data/open-sora-plan/dataset,/data/open-sora-plan/annotation/v1.1.0_HQ_part3.json
   ```

---
其中，第一个路径为数据集的根目录，第二个路径为标注文件的路径。

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

需根据实际情况修改`pretrain_t2v_model.json`和`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段。

【并行化配置参数】：

默认场景无需调整，当增大模型参数规模或者视频序列长度时，需要根据实际情况启用以下并行策略，并通过调试确定最优并行策略。

+ CP: 序列并行，当前支持Ulysess序列并行。

  - 使用场景：在视频序列（分辨率X帧数）较大时，可以开启来降低内存占用。
  
  - 使能方式：在启动脚本中设置 CP > 1，如：CP=2；
  
  - 限制条件：head数量需要能够被TP*CP整除


+ TP: 张量模型并行

  - 使用场景：模型参数规模较大时，单卡上无法承载完整的模型，通过开启TP可以降低静态内存和运行时内存。

  - 使能方式：在启动脚本中设置 TP > 1，如：TP=8

  - 限制条件：head 数量需要能够被TP*CP整除


+ SP: Megatron序列并行
  
  - 使用场景：在张量模型并行的基础上，进一步对 LayerNorm 和 Dropout 模块的序列维度进行切分，以降低动态内存。 

  - 使能方式：在 GPT_ARGS 设置 --sequence-parallel
  
  - 限制条件：前置必要条件为开启TP


+ PP：流水线并行（在研）

  目前支持将predictor模型切分流水线。在data.json文件中新增字段"pipeline_num_layers", 类型为list。该list的长度即为 pipeline rank的数量，每一个数值代表rank_i中的层数。例如，[8, 8, 8, 8]代表有4个pipeline stage， 每个容纳8个dit layers。注意list中 所有的数值的和应该和num_layers字段相等。此外，pp_rank==0的stage中除了包含dit层数以外，还会容纳text_encoder和ae，因此可以酌情减少第0个 stage的dit层数。注意保证PP模型参数配置和模型转换时的参数配置一致。

  - 使用场景：模型参数较大时候，通过流线线方式切分并行，降低内存 

  - 使能方式：使用pp时需要在运行脚本GPT_ARGS中打开以下几个参数
  
  ```shell
    PP = 4 # PP > 1 开启 
  
    --optimization-level 2 \
    --use-multiparameter-pipeline-model-parallel \
    --variable-seq-lengths \
  
    # 同时pretrain_xx_model.json中修改相应配置 
    "pipeline_num_layers": [8, 8, 8, 8],
  ```

+ VP: 虚拟流水线并行

  目前支持将predictor模型切分虚拟流水线并行。将pretrain_xxx_model.json文件中的"pipeline_num_layers"一维数组改造为两维，其中第一维表示虚拟并行的数量，二维表示流水线并行的数量，例如[[4, 4, 4, 4], [4, 4, 4, 4]]其中第一维两个数组表示vp为2, 第二维的stage个数为4表示流水线数量pp为4。

  - 使用场景：对流水线并行进行进一步切分，通过虚拟化流水线，降低空泡

  - 使能方式:如果想要使用虚拟流水线并行，需要在pretrain.t2v.sh或者prerain_i2v.sh当中修改如下变量，需要注意的是，VP仅在PP大于1的情况下生效:

  ```shell
  PP=4
  VP=4

  GPT_ARGS="
    --pipeline-model-parallel-size ${PP} \
    --virtual-pipeline-model-parallel-size ${VP} \
  ...
  ```

+ VAE-CP：VAE序列并行
  - 使用场景：视频分辨率/帧数设置的很大时，训练过程中，单卡无法完成vae的encode计算，需要开启VAE-CP
  - 使能方式：在xxx_model.json中设置vae_cp_size, vae_cp_size为大于1的整数时生效, 建议设置等于Dit部分cp_size
  - 限制条件：暂不兼容PP

【动态/固定分辨率】
- 支持使用动态分辨率或固定分辨率进行训练，默认为动态分辨率训练，如切换需修改启动脚本pretrain_xxx.sh
```shell
    # 以t2v实例，使用动态分辨率训练
    MM_DATA="./examples/opensoraplan1.3/t2v/data_dynamic_resolution.json"
    
    # 以t2v实例，使用固定分辨率训练
    MM_DATA="./examples/opensoraplan1.3/t2v/data_static_resolution.json"
```

【单机运行】

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=29501
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

【多机运行】

```shell
    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8  #每个节点的卡数
    MASTER_ADDR="your master node IP"  #都需要修改为主节点的IP地址（不能为localhost）
    MASTER_PORT=29501
    NNODES=2  #集群里的节点数，以实际情况填写,
    NODE_RANK="current node id"  #当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动预训练

t2v(文生视频):

```shell
    bash examples/opensoraplan1.3/t2v/pretrain_t2v.sh
```

i2v(图生视频):

```shell
    bash examples/opensoraplan1.3/i2v/pretrain_i2v.sh
```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，需要在每个节点准备训练数据和模型权重

---

<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作

参考上述的权重下载及转换章节，推理所需的预训练权重需要到huggingface中下载，以及参考上面的权重转换步骤进行转换。

<a id="jump5.2"></a>

#### 2. 配置参数

将准备好的权重传入到inference_t2v_model.json中，更改其中的路径，包括from_pretrained，自定义的prompt可以传入到prompt字段中

对于i2v任务，除了上述prompt，还需要将自定义的图片数据传入inference_i2v_model.json中的conditional_pixel_values_path字段中

<a id="jump5.3"></a>

#### 3. 启动推理

t2v 启动推理脚本

```shell
bash examples/opensoraplan1.3/t2v/inference_t2v.sh
```

i2v 启动推理脚本

```shell
bash examples/opensoraplan1.3/i2v/inference_i2v.sh
```

---

## 环境变量声明
CUDA_DEVICE_MAX_CONNECTIONS： 每个设备允许的最大并行硬件连接数  
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化  
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景  
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核  
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s  
PYTORCH_NPU_ALLOC_CONF： 控制缓存分配器行为  
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量  
GPUS_PER_NODE： 配置一个计算节点上使用的GPU数量  
MULTI_STREAM_MEMORY_REUSE: 配置多流内存复用是否开启，0：关闭多流内存复用，1：开启多流内存复用
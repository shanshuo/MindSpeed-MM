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
    git clone https://gitee.com/ascend/MindSpeed.git
    cd Megatron-LM
    git checkout core_r0.8.0
    cp -r megatron ../MindSpeed-MM/
    cd ..

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
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库MindSpeed core 0.8.0 使用MindSpeed master分支
    cd MindSpeed
    git checkout 82f94ca
    pip install -r requirements.txt 
    pip install -e .
    
    # 安装其余依赖库
    cd ../MindSpeed-MM
    git checkout core_r0.8.0_dev
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

MindSpeeed-MM修改了部分原始网络的结构名称，因此需要使用`convert_ckpt_to_mm.py`脚本进行转换，该脚本实现了从hugging face下载的预训练权重到到MindSpeed-MM权重的转换以及TP（Tensor Parallel）权重的切分。

首先修改 examples/opensoraplan1.3/convert_ckpt_to_mm.py 参数

    TP_SIZE = 1  # TP（Tensor Parallel）size，需要和训练脚本的TP保持一致
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

需根据实际情况修改`pretrain_t2v_model.json`、`data_xx.json`和启动脚本`pretrain_xxx.sh`中的权重和数据集路径，包括`from_pretrained`、`data_path`等字段。

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

【长序列并行】

```shell
    # 开启Ulysses长序列并行
    设置--context-parallel-algo ulysses_cp_algo
    # 开启Ring Attention长序列并行
    设置--context-parallel-algo megatron_cp_algo
    # 开启混合长序列并行
    设置--context-parallel-algo hybrid_cp_algo
    手动增添--ulysses-degree-in-cp，需要确保CP数可以被该参数整除且大于1
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

注意：使用tp训练后保存的模型无法直接进行推理，需要使用`merge_from_tp.py`脚本进行转换

首先修改 examples/opensoraplan1.3/merge_from_tp.py 参数

    dit_hg_weight_path = "mm_ckpt/open-sora-plan/pretrained-checkpoint-dit" #tp切分训练后的权重路径
    dit_mm_save_dir = "mm_ckpt/open-sora-plan/pretrained-checkpoint-dit" #合并后的权重存放路径
    MODE = "model" #选择使用model权重或者ema_model权重
---


启动脚本

    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python examples/opensoraplan1.3/merge_from_tp.py
---
<a id="jump5.2"></a>

#### 2. 配置参数

将准备好的权重传入到inference_t2v_model.json中，更改其中的路径，包括from_pretrained，自定义的prompt可以传入到prompt字段中

对于i2v任务，除了上述prompt，还需要将自定义的图片数据传入inference_i2v_model.json中的conditional_pixel_values_path字段中

<a id="jump5.3"></a>

#### 3. 启动推理
t2v 启动推理脚本

```shell
examples/opensoraplan1.3/t2v/inference_t2v.sh
```
i2v 启动推理脚本

```shell
examples/opensoraplan1.3/i2v/inference_i2v.sh
```

---

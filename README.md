  <p align="center"> <img src="sources/images/logo.png" height="103px" width="700px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
        <img alt="Badge" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

MindSpeed-MM是面向大规模分布式训练的昇腾多模态大模型套件，同时支持多模态生成及多模态理解，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、在线推理任务等特性。

注意：为保证代码安全，本仓库所有代码和配置中的trust_remote_code默认设置为False，用户需要自行全局查找该关键词并设置为True，并且确保自己下载的模型和数据的安全性

---

## MindSpeed-MM大模型方案概览

当前MindSpeed-MM支撑大模型使用功能:

* [生成类多模态大模型](#jump1) 【昇腾】【NAIE】
* [理解类多模态大模型](#jump1) 【昇腾】【NAIE】【GTS】
* [预训练/全参微调/低参微调/在线推理](./examples/) 【昇腾】【NAIE】
* 数据工程： 多模数据预处理及加载/数据分桶策略 【昇腾】
* 分布式训练： TP/PP/CP/DSP/分布式优化器/重计算 【昇腾】
* [昇腾工具链](#jump2): [Profiling采集](#jump2.1)【昇腾】

更多多模态模型持续研发中....

---

## 版本维护策略

MindSpeed-MM版本有以下五个维护阶段：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划                | 1—3 个月 | 计划特性                                                                 |
| 开发                | 3 个月   | 开发特性                                                                 |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed-MM版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                                           |

MindSpeed-MM已发布版本维护策略：

| **MindSpeed-MM版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**         | **EOL日期** |
|-----------------|-----------|--------|------------|-----------------------|-----------|
| 1.0.RC3             |  常规版本  | 维护   | 2024/09/30 | 预计2025/03/30起无维护  |           |

---

## 配套版本与支持模型

【现版本实测性能（硬件信息：Atlas 900 A2 PODc）】

下述列表中支持的模型，我们在各模型的`README`文件中提供了相应的使用说明，里面有详细的模型训练、推理、微调等流程

`模型`列中的超链接指向各模型的文件夹地址， `参数量`列中的超链接指向模型的社区资源地址

`认证`【Pass】表示已经过测试的模型，【Test】表示测试中的模型

<table>
  <a id="jump1"></a>
  <caption>MindSpeed-MM模型列表</caption>
  <thead>
    <tr>
      <th>模型任务</th>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"> 视频生成 </td>
    </tr>
    <tr>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.3">OpenSoraPlan 1.3</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 0.71 (Samples per Second) </td>
      <td> 0.73 (Samples per Second) </td>
      <td> 【昇腾】 </td>
      <td>【Pass】</td>
    </tr>
    </tbody>
</table>

---

<table>
  <caption><a href="https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/PyTorch/built-in/mm">其他已适配昇腾的多模态大模型</a></caption>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>

  </tbody>
</table>

---

<a id="jump2"></a>

## MindSpeed-MM工具库

<a id="jump2.1"></a>

### 昇腾Profiling采集工具

MindSpeed-MM集成了昇腾profiling采集工具，以提供对模型运行情况的分析。该工具能够依照配置采集模型的算子、显存等关键信息，同时支持动静态两种采集方式，协助开发者分析模型瓶颈，并可根据实际场景需求选择使用。

  具体方法见 [README](./mindspeed_mm/tools/README.md) 的profiling章节

---

## 致谢

MindSpeed-MM 由华为公司的下列部门联合贡献 ：

* 昇腾计算产品部
* 公共开发部：NAIE
* 全球技术服务部：GTS
* 计算技术开发部

感谢来自社区的每一个PR，欢迎贡献 MindSpeed-MM

---

## 安全申明

[MindSpeed MM 安全申明](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/SECURITYNOTE.md)

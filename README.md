
# E-DIS

A versatile framework for drug-target interaction prediction by considering domain specific features

# 模型介绍

预测药物-靶标相互作用(DTI)是药物发现的关键和限速步骤。传统的湿实验虽然可靠，但昂贵且耗时。深度学习方法应运而生，成为加速DTI预测的一种有前途的工具。由于化学领域的广阔，DTI预测模型通常被期望发现训练集中看不到的药物或靶标。然而，将预测性能推广到属于不同分布的新型药物靶标对是深度学习方法面临的一个挑战。在这项工作中，我们提出了一个捕获领域通用特征和领域特定特征的模型(E-DIS)，学习多种领域特征并适应分布外数据。使用混合专家作为原始数据的特定领域特征提取器，以防止在学习过程中丢失任何关键特征。多个专家在不同的领域进行训练，以捕获和对齐来自不同分布的特定领域信息，而无需访问来自未见过的领域的任何数据。E-DIS通过引入多种域特征，有效地提高了DTI预测模型的鲁棒性和泛化性

# 目录结构说明

```text
.
└─E-DIS
  ├─README.md
  ├─test.py                           # 推理网络
  ├─model.py                          # E-DIS网络
  ├─data_process.py                   # 预处理数据
  ├─dataset.py                        # 数据集处理
  ├─utils.py                          # 工具函数
  ├─EDIS_config.yaml                  # 配置文件
  └─log                               # 生成log
    ├─basic_logger.py
    ├─train_logger.py
```

# 数据集和可用模型

 本文中使用的所有数据都是公开的，可以在这里访问:
 - Davis 和 KIBA: https://github.com/hkmztrk/DeepDTA/tree/master/data
 - BindingDB 和 BioSNAP: https://github.com/peizhenbai/DrugBAN/tree/main/datasets

# 环境配置

- mindspore==2.0.0
- mindsponge-gpu==1.0.0
- mindspore-gl==0.2
- matplotlib==3.5.3
- pandas==1.3.5
- tqdm==4.65.0
- networkx==2.6.3
- numpy==1.21.6
- ipython-genutils==0.2.0
- rdkit==2023.3.1
- scikit_learn==1.0.2

# 脚本说明

## 数据预处理

```bash
python data_process.py --data_path ./data --dataset davis

option:
--data_path       # 数据路径
--dataset         # 选择数据集
```

数据处理需输入一个csv文件，文件内格式如下：

|compound_iso_smiles|target_sequence|
| --- | --- |
|SMILES|Fasta|

## 推理过程

```bash
python test.py --config ./EDIS_config.yaml --dataset davis  

option:
--config          # 配置文件路径
--dataset         # 选择数据集
```

# 参考

[1] Yang Z, Zhong W, Zhao L, et al. MGraphDTA: Deep multiscale graph neural network for explainable drug-target binding affinity prediction. Chem. Sci. 2022; 13:816–833.

# 致谢

- [MGraphDTA](https://github.com/guaguabujianle/MGraphDTA)
- [DrugBAN](https://github.com/peizhenbai/DrugBAN/)
- [DeepDTA](https://github.com/hkmztrk/DeepDTA)
- [BINDTI](https://github.com/plhhnu/BINDTI)

我们感谢所有这些开源工具的贡献者和维护者!

# E-DIS

A versatile framework for drug-target interaction prediction by considering domain specific features

# Introduction

Predicting drug-target interaction (DTI) is a critical and rate-limiting step in drug discovery. Traditional wet-lab experiments are reliable but expensive and time-consuming. Recently, deep learning has revealed itself as a new and promising tool for accelerating the DTI prediction process because its powerful performance. Due to the vast chemical space, the DTI prediction models are typically expected to discover drugs or targets that are absent from the training set. However, generalizing prediction performance to novel drug-target pairs that belong to different distributions is a challenge for deep learning methods. In this work, we propose an Ensemble of models that capture both Domain-generIc and domain-Specific features (E-DIS) to learn diversity domain features and adapt to out-of-distribution (OOD) data. We employed Mixture-of-Experts (MOE) as a domain-specific feature extractor for the raw data to prevent the loss of any crucial features by the encoder during the learning process. Multiple experts are trained on different domains to capture and align domain-specific information from various distributions without accessing any data from unseen domains. We evaluate our approach using four benchmark datasets under both in-domain and cross-domain settings and compare it with advanced approaches for solving OOD generalization problems. The results demonstrate that E-DIS effectively improves the robustness and generalizability of DTI prediction models by incorporating diversity domain features.![输入图片说明](EDIS.PNG)

# Structure

```text
.
└─E-DIS
  ├─README.md
  ├─test.py                           # inference network
  ├─model.py                          # E-DIS network
  ├─data_process.py                   # preprocessing data
  ├─dataset.py                        # processing data
  ├─utils.py                          # tool function
  ├─EDIS_config.yaml                  # configuration file
  └─log                               # generate log
    ├─basic_logger.py
    ├─train_logger.py
```

# Dataset

 All data used in this paper are publicly available and can be accessed at:
 - Davis 和 KIBA: https://github.com/hkmztrk/DeepDTA/tree/master/data
 - BindingDB 和 BioSNAP: https://github.com/peizhenbai/DrugBAN/tree/main/datasets

# Requirements

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

# Step-by-step running:

## Preprocessing data

```bash
python data_process.py --data_path ./data --dataset davis

option:
--data_path       # data path
--dataset         # choise data
```

Data processing requires input of a .csv file in the following format:

|compound_iso_smiles|target_sequence|
| --- | --- |
|SMILES|Fasta|

## Test procedure

```bash
python test.py --config ./EDIS_config.yaml --dataset davis  

option:
--config          # configuration file
--dataset         # choise data
```

# Reference

[1] Yang Z, Zhong W, Zhao L, et al. MGraphDTA: Deep multiscale graph neural network for explainable drug-target binding affinity prediction. Chem. Sci. 2022; 13:816–833.

# Acknowledgments

- [MGraphDTA](https://github.com/guaguabujianle/MGraphDTA)
- [DrugBAN](https://github.com/peizhenbai/DrugBAN/)
- [DeepDTA](https://github.com/hkmztrk/DeepDTA)

We thank all the contributors and maintainers of these open source tools!
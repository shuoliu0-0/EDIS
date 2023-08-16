# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""data process"""
import os
import stat
import argparse
from collections import OrderedDict
import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


def create_prot_csv(input_datasets):
    """extract data info from raw data and save to csv files"""

    for d_name in input_datasets:
        print('convert data from DeepDTA for ', d_name)
        fpath = 'data/' + d_name + '/'
        train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
        train_fold = [ee for e in train_fold for ee in e]
        valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
        ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
        affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
        drug_t = []
        prot_t = []
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            drug_t.append(lg)
        for t in proteins.keys():
            prot_t.append(proteins[t])
        if d_name == 'davis':
            affinity = [-np.log10(y / 1e9) for y in affinity]
        affinity = np.asarray(affinity)
        op = ['train', 'test']
        for op_t in op:
            rows, cols = np.where(np.isnan(affinity) is False)
            if op_t == 'train':
                rows, cols = rows[train_fold], cols[train_fold]
            elif op_t == 'test':
                rows, cols = rows[valid_fold], cols[valid_fold]
            with os.fdopen(os.open(f'data/{d_name}/{d_name}_{op_t}.csv', os.O_CREAT, stat.S_IWUSR), 'w') as f:
                f.write('compound_iso_smiles,target_sequence,affinity\n')
                row_list = list(range(len(rows)))
                for pair_ind in row_list:
                    ls = []
                    ls += [drug_t[rows[pair_ind]]]
                    ls += [prot_t[cols[pair_ind]]]
                    ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                    f.write(','.join(map(str, ls)) + '\n')
        print('\ndataset:', d_name)
        print('train_fold:', len(train_fold))
        print('test_fold:', len(valid_fold))
        print('len(set(drugs)),len(set(prots)):', len(set(drug_t)), len(set(prot_t)))


# nomarlize
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table.get(residue), res_pka_table.get(residue),
                     res_pkb_table.get(residue), res_pkx_table.get(residue),
                     res_pl_table.get(residue), res_hydrophobic_ph2_table.get(residue),
                     res_hydrophobic_ph7_table.get(residue)]
    return np.array(res_property1 + res_property2)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i, one_seq in enumerate(pro_seq):
        pro_hot[i,] = one_of_k_encoding(one_seq, pro_res_table)
        pro_property[i,] = residue_features(one_seq)
    return np.concatenate((pro_hot, pro_property), axis=1)


def get_nodes(graph):
    """get nodes"""
    feat = []
    for n, d in graph.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I',]]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['hybridization'] == x)
                for x in (Chem.rdchem.HybridizationType.SP, \
                          Chem.rdchem.HybridizationType.SP2, \
                          Chem.rdchem.HybridizationType.SP3)]
        h_t.append(d['num_h'])
        # 5 more
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['ImplicitValence'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = np.array([item[1] for item in feat])

    return node_attr


def get_edges(graph):
    """get edges"""
    e = {}
    for n1, n2, d in graph.edges(data=True):
        e_t = [int(d['b_type'] == x)
               for x in (Chem.rdchem.BondType.SINGLE, \
                         Chem.rdchem.BondType.DOUBLE, \
                         Chem.rdchem.BondType.TRIPLE, \
                         Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(['IsConjugated'] is False))
        e_t.append(int(d['IsConjugated'] is True))
        e[(n1, n2)] = e_t

    edge_index = np.array(list(e.keys())).transpose(1, 0)
    edge_attr = np.array(list(e.values()))
    return edge_index, edge_attr


def mol2graph(molecular):
    """mol2graph"""
    if molecular is None:
        return None
    feats = chem_feature_factory.GetFeaturesForMol(molecular)
    mol_graph = nx.DiGraph()

    # Create nodes
    for i in range(molecular.GetNumAtoms()):
        atom_i = molecular.GetAtomWithIdx(i)
        mol_graph.add_node(i,
                           a_type=atom_i.GetSymbol(),
                           a_num=atom_i.GetAtomicNum(),
                           acceptor=0,
                           donor=0,
                           aromatic=atom_i.GetIsAromatic(),
                           hybridization=atom_i.GetHybridization(),
                           num_h=atom_i.GetTotalNumHs(),

                           # 5 more node features
                           ExplicitValence=atom_i.GetExplicitValence(),
                           FormalCharge=atom_i.GetFormalCharge(),
                           ImplicitValence=atom_i.GetImplicitValence(),
                           NumExplicitHs=atom_i.GetNumExplicitHs(),
                           NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                           )

    len_feats = len(feats)
    for i in range(len_feats):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                mol_graph.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                mol_graph.nodes[n]['acceptor'] = 1

    # Read Edges
    for i in range(molecular.GetNumAtoms()):
        for j in range(molecular.GetNumAtoms()):
            e_ij = molecular.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                mol_graph.add_edge(i, j,
                                   b_type=e_ij.GetBondType(),
                                   # 1 more edge features 2 dim
                                   IsConjugated=int(e_ij.GetIsConjugated()),
                                   )

    node_attr = get_nodes(mol_graph)
    edge_index, edge_attr = get_edges(mol_graph)

    return node_attr, edge_index, edge_attr


def process_data(data_path, ligand_graph, protein_seq):
    """process_data"""
    mol_df = pd.read_csv(data_path)

    data_list = []
    for _, row in mol_df.iterrows():
        smi = row['compound_iso_smiles']
        sequence = row['target_sequence']
        label = row['affinity']

        x, edge_index, edge_attr = ligand_graph[smi]

        # caution
        x = (x - x.min()) / (x.max() - x.min())

        target_features = protein_seq[sequence]

        ##"""save data into pickle file"""
        res_t = {"x": x,
                 "y": np.array([label]),
                 "edge_index": edge_index,
                 "edge_attr": edge_attr,
                 "target": target_features,}
        data_list.append(res_t)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    m = stat.S_IWUSR | stat.S_IRUSR
    pkl_path = data_path.split('.')[0] + '.pkl'
    with os.fdopen(os.open(pkl_path, flags, m), 'wb') as f:
        pickle.dump(data_list, f)


if __name__ == '__main__':
    # choose the data to generate dataset for training or inference, support kiba or davis
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_path', default='./data/', help='data path')
    parser.add_argument('--dataset', default='davis', help='davis or kiba')

    args = parser.parse_args()
    datasets = args.dataset

    df_path = os.path.join(args.data_path, datasets, 'data_test.csv')
    df = pd.read_csv(df_path)
    compound_iso_smiles = list(df['compound_iso_smiles'])
    target_sequence = list(df['target_sequence'])
    compound_iso_smiles_set = set(compound_iso_smiles)
    target_sequences = set(target_sequence)
    graph_dict = dict()
    for smile in tqdm(compound_iso_smiles_set, total=len(compound_iso_smiles_set)):
        mol = Chem.MolFromSmiles(smile)
        g = mol2graph(mol)
        graph_dict[smile] = g

    seq_dict = dict()
    for seq in tqdm(target_sequences, total=len(target_sequences)):
        target = seq_feature(seq)

        target_feature = np.zeros((1200, 33))
        if target.shape[0] < 1200:
            target_feature[:target.shape[0]] = target
        else:
            target_feature = target[:1200]

        seq_dict[seq] = target_feature

    process_data(df_path, graph_dict, seq_dict)
from models import BINDTI
from model import MoE1
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
parser = argparse.ArgumentParser(description="BINDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='biosnap')
parser.add_argument('--split', default='random2', type=str, metavar='S', help="split task", choices=['random','random0', 'random1', 'random2', 'random3', 'random4'])
parser.add_argument('--pretrain_models1', default='./output/result/biosnap/random1/0_best_model_epoch_55.pth',type=str)
parser.add_argument('--pretrain_models2', default='./output/result/biosnap/random1/1_best_model_epoch_93.pth',type=str)
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(cfg.SOLVER.SEED)
    print(args.split)
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}')


    print("start...")
    print(f"dataset:{args.data}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./data/{args.data}'
    # dataFolder = os.path.join(dataFolder, str(args.split))


    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    print(f'train_dataset:{len(train_dataset)}')
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)


    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                                                               'drop_last':True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    # model = BINDTI(device=device, **cfg).to(device=device)
    filter_num = 32
    ensemble = [BINDTI(device=device, **cfg).to(device=device),
                MoE1(cfg["DECODER"]["IN_DIM"], num_experts=3, hidden_size=256, noisy_gating=True, 
                    k=2, **cfg).to(device)]

    trainer = Trainer(ensemble, device, training_generator, val_generator, test_generator, args.data, args.split, **cfg)
    result = trainer.train(args)

    # with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/model_architecture.txt"), "w") as wf:
    #     wf.write(str(model))
    # with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/config.txt"), "w") as wf:
    #     wf.write(str(dict(cfg)))


    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{args.data}")
    print(f'\nend...')

    return result


if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s, ")

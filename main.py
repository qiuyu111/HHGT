import os
import torch
import psutil
import time
from dataloader.cla_dataloader import NodeClassificationDataloader
from models.cla_train import NodeClassificationTask
import pytorch_lightning as pl
import yaml
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

seed = 42

TASK = {
    "node_classification": (NodeClassificationDataloader, NodeClassificationTask),
}

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def bytes_to_gb(memory_bytes):
    return memory_bytes / (1024 ** 3)

def get_trainer_model_dataloader_from_yaml(yaml_path, seed=42):
    with open(yaml_path) as f:
        settings = dict(yaml.load(f, yaml.FullLoader))
    
    set_random_seed(seed)

    DATALOADER, MODEL = TASK[settings["task"]]

    dl = DATALOADER(**settings["data"])
    model = MODEL(dl.edge_index, dl.edge_type, dl.hops, **settings["model"])
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**settings["callback"])
    trainer = pl.Trainer(callbacks=[checkpoint_callback], **settings["train"])
    return trainer, model, dl

def measure_time_and_memory(func, *args, **kwargs):
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss
    
    func(*args, **kwargs)
    
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss

    return end_time - start_time, bytes_to_gb(end_memory - start_memory)

def train(parser):
    args = parser.parse_args()
    setting_path = args.setting_path
    seed = args.seed if args.seed else 42
    trainer, model, dl = get_trainer_model_dataloader_from_yaml(setting_path, seed)
    train_time, train_memory = measure_time_and_memory(trainer.fit, model, dl)


def test(parser):
    args = parser.parse_args()
    setting_path = args.setting_path
    seed = args.seed if args.seed else 42
    trainer, model, dl = get_trainer_model_dataloader_from_yaml(setting_path, seed)
    test_time, test_memory = measure_time_and_memory(trainer.test, model, dl.test_dataloader())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_path", type=str, help="model setting file path")
    parser.add_argument("--test", action="store_true", help="test or train")
    parser.add_argument("--seed", type=int, help="random seed", default=None)
    temp_args, _ = parser.parse_known_args()
    if temp_args.test:
        test(parser)
    else:
        train(parser)
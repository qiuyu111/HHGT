import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader
from torch_sparse import coalesce
import torch_sparse
import numpy as np
import scipy.sparse as sp


class NodeClassificationDataloader(pl.LightningDataModule):
    def __init__(self, datapath, batch_size, hops, n_class, num_workers=0):
        super(NodeClassificationDataloader, self).__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.hops = hops
        self.n_class = n_class
        self.num_workers = num_workers
        self.read_data()

    def read_data(self):
        data = torch.load(self.datapath)
        train_data, test_data = data["train_data"], data["test_data"]
        self.edge_index, self.edge_type, self.feature_data = data["edge_index"], data["edge_type"], data["feature_data"]

        '''K-hop neighbor information'''
        processed_node_features = data["feature_data"]
        processed_node_features = processed_node_features[:, :self.hops+1, :, :]
        processed_node_features = processed_node_features.expand(self.n_class, processed_node_features.shape[0], processed_node_features.shape[1], processed_node_features.shape[2],processed_node_features.shape[-1])

        '''training dataset'''
        node_id_train = train_data[:,0] 
        label_train = train_data[:,1]
        node_id_traindata = processed_node_features[:,node_id_train,:,:] 
        node_id_traindata = node_id_traindata.transpose(0,1)
        self.train_dataset = TensorDataset(node_id_traindata.detach(), label_train.detach())

        '''testing dataset'''
        node_id_test = test_data[:,0] 
        label_test = test_data[:,1] 
        node_id_testdata = processed_node_features[:,node_id_test,:,: ] 
        node_id_testdata = node_id_testdata.transpose(0,1) 
        self.test_dataset = TensorDataset(node_id_testdata.detach(), label_test.detach())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)



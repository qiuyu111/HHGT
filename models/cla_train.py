from typing import Optional, Union, List

import torch
import torchmetrics
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
import numpy as np
from torch.optim import Optimizer
from models.model import HierarchicalTransformer
import os


class NodeClassificationTask(pl.LightningModule):
    def __init__(
        self,
        edge_index,
        edge_type,
        hops,
        input_dim, 
        type,
        L_hop,
        L_type,
        num_heads,
        d_model, 
        dropout_rate,
        attention_dropout_rate,
        n_class,
        use_gradient_checkpointing,
        lr,
        wd,
        train_emb_save_folder,
        test_emb_save_folder,
        train_label_save_folder,
        test_label_save_folder,
        file_name,
    ):
        super(NodeClassificationTask, self).__init__()
        self.save_hyperparameters(ignore=["edge_index", "edge_type", "processed_node_features", "N", "degree"])
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_type", edge_type)

        self.d_model = d_model
        self.n_class = n_class
        self.train_emb_save_folder = train_emb_save_folder
        self.test_emb_save_folder = test_emb_save_folder
        self.train_label_save_folder = train_label_save_folder
        self.test_label_save_folder = test_label_save_folder
        self.file_name = file_name

        self.training_embeddings = []
        self.training_labels = []
        self.testing_embeddings = []
        self.testing_labels = []
        
        self.angat = HierarchicalTransformer(type,hops,input_dim,L_hop,L_type,num_heads,d_model,dropout_rate,attention_dropout_rate,use_gradient_checkpointing)
        self.out_proj = nn.Linear(self.d_model, int(self.d_model/2))
        self.Linear1 = nn.Linear(int(self.d_model/2), self.n_class)


        self.loss = nn.CrossEntropyLoss()
        self.max_macro_F1 = -np.inf
        self.max_micro_F1 = -np.inf
        self.max_accuracy = -np.inf
        self.micro_f1_cal = torchmetrics.F1Score(num_classes=n_class, average="micro", multiclass=True)
        self.macro_f1_cal = torchmetrics.F1Score(num_classes=n_class, average="macro", multiclass=True)
        self.accuracy_cal = torchmetrics.Accuracy()

        self.w = nn.Parameter(torch.FloatTensor(n_class, d_model))
        nn.init.xavier_uniform_(self.w)

    def forward(self, batched_data):
        em = self.angat(batched_data) 
        node_em = em.transpose(0, 1) 
        return node_em
    
    def on_train_epoch_start(self) -> None:
        # Reset training_embeddings at the beginning of each epoch
        self.training_embeddings = []
        self.training_labels = []

    def on_train_epoch_end(self) -> None:
        # Save node embeddings at the end of the last epoch
        if self.current_epoch == self.trainer.max_epochs - 1:
            train_all_embeddings = np.concatenate(self.training_embeddings, axis=0)
            np.save(os.path.join(self.train_emb_save_folder, f"{self.file_name}_last_epoch"), train_all_embeddings)
            train_all_labels = np.concatenate(self.training_labels, axis=0)
            np.save(os.path.join(self.train_label_save_folder, f"{self.file_name}_last_epoch"), train_all_labels)

    def on_test_end(self) -> None:
        # Save node embeddings at the end of the last epoch
        test_all_embeddings = np.concatenate(self.testing_embeddings, axis=0)
        np.save(os.path.join(self.test_emb_save_folder, f"{self.file_name}_last_epoch"), test_all_embeddings)
        test_all_labels = np.concatenate(self.testing_labels, axis=0)
        np.save(os.path.join(self.test_label_save_folder, f"{self.file_name}_last_epoch"), test_all_labels)
    
    def evalute(self, pre, label):
        '''micro/macro_F1'''
        micro_F1 = self.micro_f1_cal(pre, label)
        macro_F1 = self.macro_f1_cal(pre, label)
        accuracy = self.accuracy_cal(pre, label)
        if self.max_micro_F1 < micro_F1:
            self.max_micro_F1 = micro_F1
            self.max_macro_F1 = macro_F1
            self.max_accuracy = accuracy
        self.log("micro-f1", micro_F1, prog_bar=True)
        self.log("macro-f1", macro_F1, prog_bar=True)
        self.log("accuracy", accuracy, prog_bar=True)
        self.micro_f1_cal.reset()
        self.macro_f1_cal.reset()
        self.accuracy_cal.reset()


    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        node_id, label = batch 
        node_id = node_id.transpose(0,1)
        embedding = self(node_id)
        logits = (embedding * self.w).sum(-1)  
        loss = self.loss(logits, label)
        self.log("loss", loss, prog_bar=True)
        
        embedding = embedding.reshape(embedding.size(0), -1)
        self.training_embeddings.append(embedding.detach().cpu().numpy())
        self.training_labels.append(label.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        node_id, label = batch 
        node_id = node_id.transpose(0,1)
        embedding = self(node_id)
        logits = (embedding * self.w).sum(-1) 
        loss = self.loss(logits, label)
        self.log("loss", loss, prog_bar=True)
        self.evalute(logits, label)

    def test_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        node_id, label = batch 
        node_id = node_id.transpose(0,1)
        embedding = self(node_id)
        logits = (embedding * self.w).sum(-1) 
        self.evalute(logits, label)

        embedding = embedding.reshape(embedding.size(0), -1)
        self.testing_embeddings.append(embedding.detach().cpu().numpy())
        self.testing_labels.append(label.detach().cpu().numpy())


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        return optimizer
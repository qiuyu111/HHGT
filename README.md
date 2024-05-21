# HHGT

Source code for paper "HHGT: Hierarchical Heterogeneous Graph Transformer for Heterogeneous Graph Representation Learning"


## Baseline links

- M2V/PTE/HIN2Vec/ComplEx/AspEm/HAN/R-GCN: https://github.com/yangji9181/HNE
- SHGP: https://github.com/kepsail/SHGP
- AGAT: https://github.com/Qidong-Liu/Aspect-Aware-Graph-Attention-Network
- HGT: https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt
- GTN/FastGTN: https://github.com/seongjunyun/Graph_Transformer_Networks
- HINormer: https://github.com/Ffffffffire/HINormer
- NAGphormer: https://github.com/JHL-HUST/NAGphormer

## Requirements

The code has been tested under Python 3.9, with the following packages installed (along with their dependencies):

- torch >= 1.9.0
- pytorch-lightning >= 1.4.4
- torchmetrics >= 0.5.0
- torch-scatter >= 2.0.9
- torch-sparse >= 0.6.12
- numpy
- pandas
- tqdm
- yaml


## Files in the folder

- **/data:** Put both acm and mag datasets under it to store the prepared data.
- **/dataloader:** Codes of the dataloader.
- **/models:** Codes of the HHGT model, semi-supervised node classification task.
- **/utils:** Codes for some utils.
- **/lightning_logs:** Store the trained model parameters, setting files, checkpoints, logs and results.
- **main.py:** The main entrance of running.
- **/records:** Store the output node embeddings for evaluation.


### Node Classification Task

**train HHGT by**

```
# train HHGT.
python main.py --setting_path *.yaml

#  for example
#  MAG dataset
python main.py --setting_path lightning_logs/mag_best/mag_settings_cla.yaml

```
The `*.yaml` is the  configuration file.

And if you want to adjust the hyperparameters of the model, you can modify it in `*.yaml`, or create a similar configuration file, and specify `--setting_path` like this:

```
python main.py --setting_path yourpath.yaml
```

Checkpoints, logs, and results during training will be stored in the directory: `./lightning_logs/version_0`


**Load the node embeddings and evaluate the results by:**

```
# test 
python cla_pre.py 

PS: Keep the configuration file unchanged during training.
```




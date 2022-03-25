import logging
from typing import Any, Optional, Union, Callable

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm.auto import trange
from transformers import get_linear_schedule_with_warmup
import dgl
import dgl.nn as dglnn

from ..backbone import BackBone, GNNClassifier
from ..basemodel import BaseTorchClassModel
from ..dataset import BaseDataset, TorchDataset
from ..utils import cross_entropy_with_probs
from ..evaluation import METRIC

logger = logging.getLogger(__name__)


class GraphModel(BaseTorchClassModel):
    def __init__(self,
                 lr: Optional[float] = 1e-4,
                 l2: Optional[float] = 0.0,
                 batch_size: Optional[int] = 128,
                 test_batch_size: Optional[int] = 512,
                 n_steps: Optional[int] = 100,
                 binary_mode: Optional[bool] = False,
                 hidden_size: int = 100,
                 block_name: str = None,
                 num_layers: int = 2,
                 ):
        super().__init__()
        if not block_name:
            block_name = "gcn"
        else:
            block_name = block_name
        self.hyperparas = {
            'lr'             : lr,
            'l2'             : l2,
            'batch_size'     : batch_size,
            'test_batch_size': test_batch_size,
            'n_steps'        : n_steps,
            'binary_mode'    : binary_mode,
            'block_name'     : block_name,
            'hidden_size'    : hidden_size,
            'num_layers'     : num_layers,
        }
        self.model: Optional[BackBone] = None

    def fit(self,
            dataset_train: BaseDataset,
            y_train: Optional[np.ndarray] = None,
            dataset_valid: Optional[BaseDataset] = None,
            y_valid: Optional[np.ndarray] = None,
            node_feats: np.ndarray = None,
            graph: dgl.DGLGraph = None,
            sample_weight: Optional[np.ndarray] = None,
            # evaluation_step: Optional[int] = 100,
            metric: Optional[Union[str, Callable]] = 'acc',
            direction: Optional[str] = 'auto',
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)

        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas

        n_steps = hyperparas['n_steps']
       
        node_feats = torch.from_numpy(node_feats).to(device)
        self.node_feats = node_feats
        self.graph = graph.to(device)
        self.dataset_valid = dataset_valid

        if y_train is None:
            y_train = dataset_train.labels
        y_train = torch.Tensor(y_train).to(device)

        if sample_weight is None:
            sample_weight = np.ones(len(dataset_train))
        sample_weight = torch.FloatTensor(sample_weight).to(device)

        n_class = dataset_train.n_class
        input_size = node_feats.shape[1]
        model = GNNClassifier(
            input_size=input_size,
            hidden_size=hyperparas['hidden_size'],
            n_class=n_class,
            num_layers=hyperparas['num_layers'],
            binary_mode=hyperparas['binary_mode'],
            model_name=hyperparas['block_name'],
        ).to(device)
        self.model = model

        optimizer = optim.Adam(model.parameters(), lr=hyperparas['lr'], weight_decay=hyperparas['l2'])

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)

        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, direction, patience, tolerance)

        history = {}
        last_step_log = {}
        
        try:
            with trange(n_steps, desc="[TRAIN] GNN Classifier", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                for step in pbar:
                    model.train()
                    optimizer.zero_grad()
                    outputs = F.softmax(model(graph, node_feats)[dataset_train.node_id], dim=-1)
                    target = y_train
                    loss = cross_entropy_with_probs(outputs, target, reduction='none')
                    loss = torch.mean(loss * sample_weight)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if valid_flag:
                        metric_value, early_stop_flag, info = self._valid_step(step)
                        if early_stop_flag:
                            logger.info(info)
                            break
                        if step == n_steps:
                            history[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history[step])

                    last_step_log['loss'] = loss.item()
                    # pbar.update()
                    pbar.set_postfix(ordered_dict=last_step_log)

        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        self._finalize()

        return history


    def _calc_valid_metric(self, **kwargs):
        # with autocast(enabled=get_amp_flag()):
        probas = self.predict_proba(self.dataset_valid.node_id, self.node_feats, self.graph, **kwargs)
        return self.metric_fn(self.y_valid, probas)

    def test(self, dataset: BaseDataset, metric_fn: Union[Callable, str], y_true: Optional[np.ndarray] = None, **kwargs):
        if isinstance(metric_fn, str):
            metric_fn = METRIC[metric_fn]
        if y_true is None:
            y_true = np.array(dataset.labels)
        probas = self.predict_proba(dataset.node_id, self.node_feats, self.graph, **kwargs)
        return metric_fn(y_true, probas)

    @torch.no_grad()
    def predict_proba(self, node_id, node_feats, graph: dgl.DGLGraph, device: Optional[torch.device] = None, **kwargs: Any):
        if device is not None:
            model = self.model.to(device)
        else:
            model = self.model
        model.eval()
        
        probas = []
        output = model(graph, node_feats)[node_id]
        if output.shape[1] == 1:
            output = torch.sigmoid(output)
            proba = torch.cat([1 - output, output], -1)
        else:
            proba = F.softmax(output, dim=-1)
        probas.append(proba.cpu().detach().numpy())

        return np.vstack(probas)

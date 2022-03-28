import logging
from typing import Any, Optional, Union

import numpy as np
from tqdm.auto import tqdm
import dgl
import scipy.sparse as sp
# from snorkel.labeling.model import LabelModel

from ..basemodel import BaseLabelModel
from ..dataset import BaseDataset, GraphDataset, normalize_adj
from ..dataset.utils import check_weak_labels

logger = logging.getLogger(__name__)

def build_graph_(dataset: GraphDataset):
    """
        Build a graph based on weak labels and origin graph
    """
    L = check_weak_labels(dataset)
    n_rules = L.shape[-1]
    graph = dataset.load_graph()
    node_ids = np.array(dataset.node_id)
    src, target = graph.edges()
    src, target = src.cpu().numpy(), target.cpu().numpy()
    max_node_ids = np.max(graph.nodes().cpu().numpy()) + 1
    n_class = dataset.n_class
    weak_nodes = np.array([(k + j * n_rules + max_node_ids)  for j in range(n_class) for k in range(n_rules)])
    weak_labels = np.array([j for j in range(n_class) for k in range(n_rules)])
    # rule_mask = np.zeros_like(L, dtype=np.bool)
    src_weak = [src]
    target_weak = [target]
    for rule in range(n_rules):
        labels = L[:, rule]
        mask = (labels != -1)
        labels = labels[mask]
        src_weak.append(weak_nodes[rule + n_rules * labels])
        target_weak.append(node_ids[mask])
    src_weak = np.concatenate(src_weak)
    target_weak = np.concatenate(target_weak)

    max_node_idx = np.where(weak_nodes == np.max(src_weak))[0][-1]
    graph_weak = dgl.graph((src_weak, target_weak))
    return weak_nodes[: max_node_idx + 1], weak_labels[: max_node_idx + 1], graph_weak
    
    
class LPA(BaseLabelModel):
    def __init__(self,
                 alpha: Optional[float] = 1.0,
                 lpa_iters: Optional[int] = 50,
                 **kwargs: Any):
        super().__init__()
        self.hyperparas = {
            'alpha'     : alpha,
            'lpa_iters': lpa_iters
        }

    def fit(self,
            dataset_train: GraphDataset,
            n_class: Optional[int] = None,
            **kwargs: Any):
        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas
        if isinstance(dataset_train, BaseDataset):
            if n_class is not None:
                assert n_class == dataset_train.n_class
            else:
                n_class = dataset_train.n_class
        self.n_class = n_class or int(np.max(check_weak_labels(dataset_train))) + 1
        weak_nodes, weak_labels, weak_graph = build_graph_(dataset_train)
        adj = normalize_adj(weak_graph)
        # adj = sp.coo_matrix(weak_graph.adj().to_dense())
        labels = np.zeros([adj.shape[0] , n_class])
        labels[weak_nodes] = np.eye(n_class)[weak_labels]
    
        alpha = hyperparas['alpha']
        y = labels.copy()
        for _ in tqdm(range(hyperparas['lpa_iters'])):
            y = alpha * (adj.transpose() @ y) + (1 - alpha) * labels
            y = y.clip(0, 1)
        self.results = y

    def predict_proba(self, dataset: GraphDataset, **kwargs: Any) -> np.ndarray:
        results = self.results
        node_ids = dataset.node_id
        correct = np.sum(np.argmax(results[node_ids], axis=-1) == dataset.labels)
        return correct/len(node_ids)

    def test(self, dataset: GraphDataset, **kwargs: Any):
        return self.predict_proba(dataset)

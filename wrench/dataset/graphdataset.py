import json
from pathlib import Path
from typing import Any, List, Optional, Union

import os
import numpy as np
import torch
import dgl
import logging
from tqdm.auto import tqdm
from torchvision.datasets.folder import pil_loader
from .dataset import NumericDataset, TextDataset
from .basedataset import BaseDataset
from .utils import bag_of_words_extractor, tf_idf_extractor, sentence_transformer_extractor, \
    bert_text_extractor, bert_relation_extractor, image_feature_extractor


logger = logging.getLogger(__name__)

def extract_node_features(graph : dgl.DGLGraph, node_features : np.ndarray, node_id : np.ndarray, init : str = "uniform", dense : bool = False):

    assert len(node_features) == len(node_id)
    if dense:
        node_features = dense_to_sparse(node_features)
    nodes =  graph.nodes().cpu().numpy()
    unlabeled_nodes = list(set(nodes) - set(node_id))
    if len(unlabeled_nodes) == 0:
        return node_features
    feature_shape = np.concatenate([nodes.shape[0]], node_features.shape[1:])
    unlabeled_feature_shape = np.concatenate([len(unlabeled_nodes)], node_features.shape[1:])
    features = np.empty(feature_shape, dtype=np.float32)
    features[node_id] =  node_features

    if init == "normal":
        features[unlabeled_nodes] = np.random.normal(size=unlabeled_feature_shape)
    elif init == "uniform":
        features[unlabeled_nodes] = np.random.uniform(size=unlabeled_feature_shape)

    return features

def dense_to_sparse(node_features : np.ndarray):
    max_feat = np.nanmax(node_features)
    features = np.zeros([node_features.shape[0], int(max_feat) + 1], dtype=np.float32)
    nan_map = np.invert(np.isnan(node_features))
    for idx, feature in enumerate(features):
        feature_indices = node_features[idx][nan_map[idx]].astype(np.int)
        features[idx][feature_indices] = 1
    return features

class GraphDataset(BaseDataset):
    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        import dgl
        super().__init__(path, split, feature_cache_name, **kwargs)
        if self.path is not None:
            self.graph_path = self.path / f'graph.bin'
            # self.graph = dgl.load_graphs(str(self.graph_path))

    def load(self, path: str, split: str):
        super().load(self.path, self.split)
        data_path = path / f'{split}.json'
        self.node_id = []
        data = json.load(open(data_path, 'r'))
        for i, item in tqdm(data.items()):
            self.node_id.append(item["data"]["node_id"])
        return self

    def create_subset(self, idx: List[int]):
        dataset = super().create_subset(idx)
        dataset.node_id = []
        for i in idx:
            dataset.node_id.append(self.node_id[i])
        return dataset
    
    def load_graph(self):
        self.graph = dgl.load_graphs(str(self.graph_path))[0]
        return self.graph[0]

class GraphNumericDataset(GraphDataset, NumericDataset):
    """Data class for numeric dataset."""
    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        GraphDataset.__init__(self, path, split, feature_cache_name, **kwargs)


class GraphTextDataset(GraphDataset, TextDataset):
    """Data class for text graph node classification dataset."""

    def __init__(self,
                 path: str = None,
                 split: Optional[str] = None,
                 feature_cache_name: Optional[str] = None,
                 **kwargs: Any) -> None:
        GraphDataset.__init__(self, path, split, feature_cache_name,  **kwargs)
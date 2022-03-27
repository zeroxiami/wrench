import logging
import torch
import numpy as np
from wrench.dataset import load_dataset, extract_node_features
from wrench.logging import LoggingHandler
from wrench.labelmodel import Snorkel, LPA
from wrench.endmodel import GraphModel
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda:1')

#### Load dataset
dataset_path = './datasets/'
data = 'facebook_large'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    device=device,
    # extract_fn='bert', # extract bert embedding
    # model_name='bert-base-cased',
    # cache_name='bert'
)

#### Run label model: LPA
lpa = LPA(
    alpha=1.0,
    lpa_iters=100,
)
lpa.fit(
    dataset_train=train_data,
)
acc = lpa.test(test_data)
logger.info(f'LPA test acc: {acc}')
#### Run label model: Snorkel
label_model = Snorkel(
    lr=0.01,
    l2=0.0,
    n_epochs=10
)
label_model.fit(
    dataset_train=train_data,
    dataset_valid=valid_data
)
acc = label_model.test(test_data, 'acc')
logger.info(f'label model test acc: {acc}')


# model = RGCN(n_hetero_features, 20, n_user_classes, hetero_graph.etypes)
graph = train_data.load_graph().to(device)
node_feats = np.concatenate([train_data.features, valid_data.features, test_data.features])
node_id = np.concatenate([train_data.node_id, valid_data.node_id, test_data.node_id])
# gt = np.concatenate([train_data.labels, valid_data.labels, test_data.labels])
node_feats = extract_node_features(graph=graph, node_features=node_feats, node_id=node_id, dense=True)
# node_feats = torch.from_numpy(node_feats).to(device)
# gt = torch.from_numpy(gt).to(device)
# node_features =  {'node_id': user_feats}
#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
aggregated_hard_labels = label_model.predict(train_data)
aggregated_soft_labels = label_model.predict_proba(train_data)

#### Run end model: MLP
model = GraphModel(
    n_steps=50,
    lr=1e-3,
    block_name='sage',
)
model.fit(
    dataset_train=train_data,
    node_feats=node_feats,
    graph=graph,
    y_train=aggregated_hard_labels,
    dataset_valid=valid_data,
    metric='acc',
    patience=100,
    device=device,
)
acc = model.test(test_data, 'acc')
logger.info(f'end model (GNN) test acc: {acc}')

# item_feats = hetero_graph.nodes['item'].data['feature']


# Run end model: RGCN
# for epoch in range(1000):
#     model.train()
#     # forward propagation by using all nodes and extracting the user embeddings
#     logits = model(hetero_graph, node_features)['user']
#     # compute loss
#     loss = F.cross_entropy(logits[train_mask], labels[train_mask])
#     # Compute validation accuracy.  Omitted in this example.
#     # backward propagation
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     print(loss.item())
# n_features = node_feats.shape[1]
# n_labels = int(max(gt) + 1)
# graph = graph
# # Run end model: GraphSage
# model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels).to(device)
# opt = torch.optim.Adam(model.parameters())
# schedulerC = torch.optim.lr_scheduler.MultiStepLR(opt, [30, 80])
# criterion = nn.CrossEntropyLoss().to(device)
# for epoch in range(100):
#     model.train()
#     # forward propagation by using all nodes
#     preds = model(graph, node_feats)
#     preds = F.softmax(preds[train_node_id], dim=-1)
#     # compute loss
#     loss = criterion(preds, aggregated_hard_labels)
#     # compute validation accuracy
#     acc = evaluate(model, graph, node_feats, gt, valid_node_id)
#     # backward propagation
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     schedulerC.step()
#     if epoch % 10 == 0:
#         print("Epoch {} ACC: {}".format(epoch, acc))
#         print(loss.item())
# acc = evaluate(model, graph, node_feats, gt, test_node_id)
# print("Test acc: {}".format(acc))

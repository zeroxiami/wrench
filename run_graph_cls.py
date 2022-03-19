import logging
import torch
import numpy as np
from wrench.dataset import load_dataset, extract_node_features
from wrench.logging import LoggingHandler
from wrench.labelmodel import Snorkel
from wrench.endmodel import EndClassifierModel
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

device = torch.device('cuda:1')

#### Load dataset
dataset_path = './datasets/'
data = 'git_ml'
train_data, valid_data, test_data = load_dataset(
    dataset_path,
    data,
    extract_feature=True,
    device=device,
    # extract_fn='bert', # extract bert embedding
    # model_name='bert-base-cased',
    # cache_name='bert'
)

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
gt = np.concatenate([train_data.labels, valid_data.labels, test_data.labels])
node_feats = extract_node_features(graph=graph, node_features=node_feats, node_id=node_id, dense=True)
node_feats = torch.from_numpy(node_feats).to(device)
gt = torch.from_numpy(gt).to(device)
# node_features =  {'node_id': user_feats}
#### Filter out uncovered training data
train_data = train_data.get_covered_subset()
aggregated_hard_labels = torch.from_numpy(label_model.predict(train_data)).to(device)
aggregated_soft_labels = torch.from_numpy(label_model.predict_proba(train_data)).to(device)
train_node_id = train_data.node_id
valid_node_id = valid_data.node_id
test_node_id = test_data.node_id


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
n_features = node_feats.shape[1]
n_labels = int(max(gt) + 1)
# Run end model: GraphSage
model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels).to(device)
opt = torch.optim.Adam(model.parameters())
schedulerC = torch.optim.lr_scheduler.MultiStepLR(opt, [30, 80])
criterion = nn.CrossEntropyLoss().to(device)
for epoch in range(100):
    model.train()
    # forward propagation by using all nodes
    preds = model(graph, node_feats)
    preds = F.softmax(preds[train_node_id], dim=-1)
    # compute loss
    loss = criterion(preds, aggregated_hard_labels)
    # compute validation accuracy
    acc = evaluate(model, graph, node_feats, gt, valid_node_id)
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    schedulerC.step()
    if epoch % 10 == 0:
        print("Epoch {} ACC: {}".format(epoch, acc))
        print(loss.item())
acc = evaluate(model, graph, node_feats, gt, test_node_id)
print("Test acc: {}".format(acc))

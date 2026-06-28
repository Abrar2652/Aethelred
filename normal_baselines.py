# -*- coding: utf-8 -*-
"""
Normal (non-certified) GNN baselines used for Table 1-style comparisons.
"""

import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from datasets.dataset_loader import load_graph_data
from datasets.dataset_loader import load_node_data
from gnn import GraphGAT
from gnn import GraphGCN
from gnn import GraphGSAGE
from gnn import NodeGAT
from gnn import NodeGCN
from gnn import NodeGSAGE
from utils import evaluate
from utils import train_model
from utils import store_checkpoint


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_node_model(gnn, num_x, num_labels):
    if gnn == "GCN":
        return NodeGCN(num_x, num_labels)
    if gnn == "GSAGE":
        return NodeGSAGE(num_x, num_labels)
    if gnn == "GAT":
        return NodeGAT(num_x, num_labels)
    raise ValueError(f"Unsupported node GNN: {gnn}")


def _build_graph_model(gnn, num_x, num_labels):
    if gnn == "GCN":
        return GraphGCN(num_x, num_labels)
    if gnn == "GSAGE":
        return GraphGSAGE(num_x, num_labels)
    if gnn == "GAT":
        return GraphGAT(num_x, num_labels)
    raise ValueError(f"Unsupported graph GNN: {gnn}")


def _load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _checkpoint_path(task, gnn, dataset):
    return f"./checkpoints/normal_{task}/{gnn}/{dataset}/best_model"


def _predict_graphs(model, graphs, device):
    outputs = []
    loader = DataLoader(list(graphs), batch_size=256, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs.append(model(batch.x, batch.edge_index, batch=batch.batch).cpu())

    if not outputs:
        out_dim = model.lin.out_features
        return torch.zeros((0, out_dim), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def _train_node_model(model, x, edge_index, labels, train_mask, val_mask, test_mask, train_args):
    device = _device()
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Match the original PGNNCert node baseline training path exactly.
    ref_args = dict(train_args)
    ref_args["paper"] = f"normal_node/{train_args['paper']}"
    train_model(model, edge_index, x, labels, train_mask, val_mask, test_mask, ref_args)


def _train_graph_model(model, graphs, labels, train_mask, val_mask, test_mask, train_args):
    device = _device()
    model = model.to(device)
    labels_t = torch.tensor(labels, dtype=torch.long)

    train_graphs = graphs[train_mask]
    val_graphs = graphs[val_mask]
    test_graphs = graphs[test_mask]
    train_labels = labels_t[train_mask]
    val_labels = labels_t[val_mask]
    test_labels = labels_t[test_mask]
    train_loader = DataLoader(list(train_graphs), batch_size=train_args["batch_size"], shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(train_args["epochs"]):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = criterion(out, batch.y.view(-1).to(torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args["clip_max"])
            optimizer.step()
            running_loss += loss.item() * batch.num_graphs

        train_out = _predict_graphs(model, train_graphs, device)
        val_out = _predict_graphs(model, val_graphs, device)
        test_out = _predict_graphs(model, test_graphs, device)
        train_acc = evaluate(train_out, train_labels)
        val_acc = evaluate(val_out, val_labels)
        test_acc = evaluate(test_out, test_labels)
        avg_loss = running_loss / max(len(train_graphs), 1)

        print(
            f"Epoch: {epoch}, train_acc: {train_acc:.4f}, "
            f"val_acc: {val_acc:.4f}, train_loss: {avg_loss:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(
                f"normal_graph/{train_args['paper']}",
                train_args["dataset"],
                model,
                train_acc,
                val_acc,
                test_acc,
            )

        if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
            break


def run_normal_node(dataset, gnn, train_args, retrain=False):
    _set_seed(train_args.get("seed", 42))
    device = _device()

    data, num_x, num_labels = load_node_data(dataset)
    x = torch.as_tensor(data.x, dtype=torch.float32)
    edge_index = torch.as_tensor(data.edge_index, dtype=torch.int64)
    labels = torch.as_tensor(data.y, dtype=torch.long)

    model = _build_node_model(gnn, num_x, num_labels)
    path = _checkpoint_path("node", gnn, dataset)

    if (not retrain) and os.path.exists(path):
        _load_checkpoint(path, model, device)
    else:
        _train_node_model(
            model,
            x,
            edge_index,
            labels,
            data.train_mask,
            data.val_mask,
            data.test_mask,
            train_args,
        )
        _load_checkpoint(path, model, device)

    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device)).cpu()
    return evaluate(out[data.test_mask], labels[data.test_mask])


def run_normal_graph(dataset, gnn, train_args, retrain=False):
    _set_seed(train_args.get("seed", 42))
    device = _device()

    graphs, num_x, num_labels, mask_split, labels = load_graph_data(dataset)
    train_mask, val_mask, test_mask = mask_split

    model = _build_graph_model(gnn, num_x, num_labels)
    path = _checkpoint_path("graph", gnn, dataset)

    if (not retrain) and os.path.exists(path):
        _load_checkpoint(path, model, device)
    else:
        _train_graph_model(model, graphs, labels, train_mask, val_mask, test_mask, train_args)
        _load_checkpoint(path, model, device)

    labels_t = torch.tensor(labels, dtype=torch.long)
    test_out = _predict_graphs(model, graphs[test_mask], device)
    return evaluate(test_out, labels_t[test_mask])

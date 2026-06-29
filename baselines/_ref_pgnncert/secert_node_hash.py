# -*- coding: utf-8 -*-
"""
SECert-N: Shared-Encoder Certified Defense (Node-Centric Variant)

Same innovations as SECert-E but with node-centric graph partitioning:
  - Hash NODES (not edges) to assign them to subgraphs
  - Shared GNN backbone + per-subgraph heads
  - Margin-boosting training loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import numpy as np
from torch_geometric.data import Data

from gnn import SharedNodeBackbone, SharedGraphBackbone, SubgraphHead
from utils import evaluate, store_secert_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"


class HashAgent:
    """Node-centric hash agent — identical to PGNNCert-N for fair comparison."""

    def __init__(self, h="md5", T=30):
        self.T = T
        self.h = h

    def hash_node(self, u):
        hexstring = hex(u).encode()
        if self.h == "md5":
            hash_device = hashlib.md5()
        elif self.h == "sha1":
            hash_device = hashlib.sha1()
        elif self.h == "sha256":
            hash_device = hashlib.sha256()
        hash_device.update(hexstring)
        I = int(hash_device.hexdigest(), 16) % self.T
        return I

    def generate_node_subgraphs(self, edge_index, x, y):
        subgraphs = []
        original = edge_index
        V = x.shape[0]

        for i in range(self.T):
            subgraphs.append(Data(x=x, y=y, edge_index=[]))

        for i in range(len(original[0])):
            u = original[0, i]
            v = original[1, i]
            I = self.hash_node(u)
            subgraphs[I].edge_index.append([u, v])

        new_subgraphs = []
        for i in range(self.T):
            if len(subgraphs[i].edge_index) == 0:
                continue
            subgraphs[i].edge_index = torch.tensor(
                subgraphs[i].edge_index, dtype=torch.int64
            ).transpose(1, 0)
            new_subgraphs.append(subgraphs[i])
        return new_subgraphs

    def generate_graph_subgraphs(self, edge_index, x, y):
        subgraphs = []
        original = edge_index
        zerox = torch.zeros(x[0].size()).reshape(1, -1)
        V = x.shape[0]
        mappings = []

        for i in range(self.T):
            subgraphs.append(Data(x=zerox, y=y, edge_index=[[0, 0]]))

        for i in range(x.shape[0]):
            I = self.hash_node(i)
            mappings.append(subgraphs[I].x.shape[0])
            subgraphs[I].x = torch.cat((subgraphs[I].x, x[i].reshape(1, -1)), dim=0)
            subgraphs[I].edge_index.append([mappings[i], 0])

        for i in range(len(original[0])):
            u = original[0, i]
            v = original[1, i]
            I = self.hash_node(u)
            if self.hash_node(v) == I:
                subgraphs[I].edge_index.append([mappings[u], mappings[v]])

        for i in range(self.T):
            subgraphs[i].edge_index = torch.tensor(
                subgraphs[i].edge_index, dtype=torch.int64
            ).transpose(1, 0)
        return subgraphs


class SECertNodeClassifier(nn.Module):
    """SECert Node-Centric Node Classifier with shared backbone."""

    def __init__(self, Hasher, edge_index, x, y, train_mask, val_mask, test_mask,
                 num_x, num_labels, GNN="GCN", lambda_margin=0.1):
        super().__init__()
        self.Hasher = Hasher
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.edge_index = edge_index
        self.x = x
        self.y = y.to(self.device)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.train_mask_device = train_mask.to(self.device)
        self.val_mask_device = val_mask.to(self.device)
        self.test_mask_device = test_mask.to(self.device)
        self.num_labels = num_labels
        self.T = Hasher.T
        self.lambda_margin = lambda_margin

        hidden_size = 20
        self.backbone = SharedNodeBackbone(num_x, hidden_size, conv_type=GNN).to(self.device)
        embedding_size = self.backbone.embedding_size

        self.heads = nn.ModuleList([
            SubgraphHead(embedding_size, num_labels, use_adapter=False).to(self.device)
            for _ in range(self.T)
        ])

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def _compute_margin_loss(self, votes, true_labels):
        batch_size = votes.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)
        true_votes = votes[torch.arange(batch_size), true_labels]
        masked_votes = votes.clone()
        masked_votes[torch.arange(batch_size), true_labels] = -1e9
        runner_up_votes = masked_votes.max(dim=1)[0]
        margin = true_votes - runner_up_votes
        hinge_loss = F.relu(2.0 - margin).mean()
        return hinge_loss

    def train_model(self, train_args):
        subgraphs = self.Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        optimizer = torch.optim.Adam(self.parameters(), lr=train_args["lr"])
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(train_args["epochs"]):
            self.backbone.train()
            for h in self.heads:
                h.train()

            optimizer.zero_grad()
            loss_ce = torch.zeros(1, device=self.device)

            all_preds = []
            for i in range(len(subgraphs)):
                emb = self.backbone(subgraphs[i].x.to(self.device),
                                    subgraphs[i].edge_index.to(self.device))
                out = self.heads[i](emb)
                loss_ce += criterion(out[self.train_mask_device], self.y[self.train_mask_device])
                all_preds.append(out)

            # Margin loss
            loss_margin = torch.zeros(1, device=self.device)
            if self.lambda_margin > 0 and len(all_preds) > 0:
                train_count = self.train_mask.sum().item()
                soft_votes = torch.zeros(train_count, self.num_labels, device=self.device)
                for i in range(len(all_preds)):
                    probs = F.softmax(all_preds[i][self.train_mask_device].detach(), dim=1)
                    soft_votes += probs
                loss_margin = self._compute_margin_loss(soft_votes, self.y[self.train_mask_device])

            loss = loss_ce + self.lambda_margin * loss_margin
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=train_args["clip_max"])
            optimizer.step()

            with torch.no_grad():
                out_train, _ = self.vote(self.train_mask)
                out_val, _ = self.vote(self.val_mask)
                out_test, _ = self.vote(self.test_mask)
                train_acc = evaluate(out_train.to(self.device), self.y[self.train_mask_device])
                val_acc = evaluate(out_val.to(self.device), self.y[self.val_mask_device])
                test_acc = evaluate(out_test.to(self.device), self.y[self.test_mask_device])

            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, "
                  f"train_loss: {loss.item():.4f}")

            if val_acc > best_val_acc:
                print("Val improved")
                best_val_acc = val_acc
                best_epoch = epoch
                store_secert_checkpoint(
                    "secert_n/" + train_args["paper"],
                    train_args["dataset"] + "/{}".format(self.T),
                    self.state_dict(), train_acc, val_acc, test_acc
                )

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break

    def test(self):
        out_test, M = self.vote(self.test_mask)
        test_acc = evaluate(out_test.to(self.device), self.y[self.test_mask_device])
        return test_acc, M

    def vote(self, mask):
        subgraphs = self.Hasher.generate_node_subgraphs(self.edge_index, self.x, self.y)
        V_test = self.x[mask].shape[0]
        votes = torch.zeros((V_test, self.num_labels))
        mask_device = mask.to(self.device)

        self.backbone.eval()
        for i in range(len(subgraphs)):
            self.heads[i].eval()
            emb = self.backbone(subgraphs[i].x.to(self.device),
                                subgraphs[i].edge_index.to(self.device))
            out = self.heads[i](emb)
            preds = out[mask_device].argmax(dim=1).cpu()
            for j in range(V_test):
                votes[j, preds[j]] += 1

        vote_label = votes.argmax(dim=1)
        M = torch.zeros(V_test)
        for i in range(V_test):
            votes[i, vote_label[i]] = -votes[i, vote_label[i]]
        second_label = votes.argmax(dim=1)
        for i in range(V_test):
            votes[i, vote_label[i]] = -votes[i, vote_label[i]]
        for i in range(V_test):
            if vote_label[i] > second_label[i]:
                M[i] = (votes[i, vote_label[i]] - votes[i, second_label[i]] - 1) // 2
            else:
                M[i] = (votes[i, vote_label[i]] - votes[i, second_label[i]]) // 2
        return votes, M


class SECertGraphClassifier(nn.Module):
    """SECert Node-Centric Graph Classifier with shared backbone."""

    def __init__(self, Hasher, graphs, labels, train_mask, val_mask, test_mask,
                 num_x, num_labels, GNN="GCN", lambda_margin=0.1):
        super().__init__()
        self.Hasher = Hasher
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_labels = num_labels
        self.graphs = graphs
        self.labels = torch.tensor(labels)
        self.T = Hasher.T
        self.lambda_margin = lambda_margin

        hidden_size = 32
        self.backbone = SharedGraphBackbone(num_x, hidden_size, conv_type=GNN).to(self.device)
        embedding_size = self.backbone.embedding_size

        self.heads = nn.ModuleList([
            SubgraphHead(embedding_size, num_labels, use_adapter=False).to(self.device)
            for _ in range(self.T)
        ])

        self.subgraphsX = [[] for _ in range(self.T)]
        self.subgraphsE = [[] for _ in range(self.T)]
        for i in range(len(graphs)):
            subgraphs = self.Hasher.generate_graph_subgraphs(
                graphs[i].edge_index, graphs[i].x, graphs[i].y
            )
            for j in range(self.T):
                self.subgraphsX[j].append(subgraphs[j].x)
                self.subgraphsE[j].append(subgraphs[j].edge_index)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])

    def enlarge_dataset(self, graphs):
        new_graphs = []
        ys = []
        for i in range(len(graphs)):
            subgraphs = self.Hasher.generate_graph_subgraphs(
                graphs[i].edge_index, graphs[i].x, graphs[i].y
            )
            new_graphs.append([])
            ys.append([])
            for j in range(self.T):
                new_graphs[-1].append((subgraphs[j].x, subgraphs[j].edge_index))
                ys[-1].append(subgraphs[j].y)
            ys[-1] = torch.tensor(ys[-1], dtype=subgraphs[0].y.dtype)
        return new_graphs, ys

    def train_model(self, train_args):
        optimizer = torch.optim.Adam(self.parameters(), lr=train_args["lr"])
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0

        train_graphs = self.graphs[self.train_mask]
        entrain_graphs, ys = self.enlarge_dataset(train_graphs)

        for epoch in range(train_args["epochs"]):
            optimizer.zero_grad()
            loss = torch.zeros(1, device=self.device)

            self.backbone.train()
            for h in self.heads:
                h.train()

            for i in range(len(entrain_graphs)):
                out = torch.zeros((self.T, self.num_labels), device=self.device)
                for j in range(self.T):
                    x_j, edge_index_j = entrain_graphs[i][j]
                    x_j = x_j.to(self.device)
                    edge_index_j = edge_index_j.to(self.device) if edge_index_j is not None else None
                    emb = self.backbone(x_j, edge_index_j)
                    out[j] = self.heads[j](emb)
                loss += criterion(out, ys[i].to(self.device, dtype=torch.long))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=train_args["clip_max"])
            optimizer.step()

            with torch.no_grad():
                out_train, _ = self.vote(self.train_mask)
                out_val, _ = self.vote(self.val_mask)
                train_acc = evaluate(out_train, self.labels[self.train_mask])
                val_acc = evaluate(out_val, self.labels[self.val_mask])

            print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, "
                  f"train_loss: {loss.item():.4f}")

            if val_acc == best_val_acc and train_acc > best_train_acc:
                print("Train improved")
                best_train_acc = train_acc
                best_epoch = epoch
                store_secert_checkpoint(
                    "secert_n/" + train_args["paper"],
                    train_args["dataset"] + "/{}".format(self.T),
                    self.state_dict(), train_acc, val_acc, 0.0
                )

            if val_acc > best_val_acc:
                print("Val improved")
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                store_secert_checkpoint(
                    "secert_n/" + train_args["paper"],
                    train_args["dataset"] + "/{}".format(self.T),
                    self.state_dict(), train_acc, val_acc, 0.0
                )

            if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
                break

    def test(self):
        out_test, M = self.vote(self.test_mask)
        test_acc = evaluate(out_test, self.labels[self.test_mask])
        return test_acc, M

    def vote(self, mask):
        G_test = len(self.graphs[mask])
        idxs = np.array([i for i in range(len(self.graphs))])
        test_id = idxs[mask]

        votes = torch.zeros((G_test, self.num_labels))

        self.backbone.eval()
        for j in range(self.T):
            self.heads[j].eval()
            out = torch.zeros((test_id.shape[0], self.num_labels), device=self.device)
            for i in range(test_id.shape[0]):
                x_j = self.subgraphsX[j][test_id[i]].to(self.device)
                edge_index_j = self.subgraphsE[j][test_id[i]]
                edge_index_j = edge_index_j.to(self.device) if edge_index_j is not None else None
                emb = self.backbone(x_j, edge_index_j)
                out[i] = self.heads[j](emb)
            preds = out.argmax(dim=1).cpu()
            for i in range(preds.shape[0]):
                votes[i, preds[i]] += 1

        vote_label = votes.argmax(dim=1)
        M = torch.zeros(G_test)
        for i in range(G_test):
            votes[i, vote_label[i]] = -votes[i, vote_label[i]]
        second_label = votes.argmax(dim=1)
        for i in range(G_test):
            votes[i, vote_label[i]] = -votes[i, vote_label[i]]
        for i in range(G_test):
            if vote_label[i] > second_label[i]:
                M[i] = (votes[i, vote_label[i]] - votes[i, second_label[i]] - 1) // 2
            else:
                M[i] = (votes[i, vote_label[i]] - votes[i, second_label[i]]) // 2
        return votes, M

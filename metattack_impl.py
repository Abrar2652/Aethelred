# -*- coding: utf-8 -*-
"""
Standalone MetaAttack implementation ported from:
  https://github.com/ChandlerBang/pytorch-gnn-meta-attack
  (Zügner & Günnemann, ICLR 2019)

Self-contained: no DeepRobust dependency.
All .cuda() calls replaced with explicit device handling.
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
from tqdm import tqdm


# ======================================================================
# Utility functions (ported from utils.py)
# ======================================================================

def _normalize_adj_tensor(adj, device):
    """Symmetric normalisation: D^{-1/2} (A+I) D^{-1/2}"""
    mx = adj + torch.eye(adj.shape[0], device=device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx @ r_mat_inv
    return mx


def _accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)


def _unravel_index(index, array_shape):
    rows = index // array_shape[1]
    cols = index % array_shape[1]
    return rows, cols


def _degree_sequence_log_likelihood(degree_sequence, d_min):
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    alpha = _compute_alpha(n, sum_log_degrees, d_min)
    ll = _compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


def _compute_alpha(n, sum_log_degrees, d_min):
    return 1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))


def _compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    return (n * torch.log(alpha)
            + n * alpha * torch.log(d_min)
            + (alpha + 1) * sum_log_degrees)


def _update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()
    sum_log_degrees_after = (sum_log_degrees_before
                             - torch.log(torch.clamp(d_old_in_range, min=1)).sum(1)
                             + torch.log(torch.clamp(d_new_in_range, min=1)).sum(1))
    new_n = (n_old - (old_in_range != 0).sum(1) + (new_in_range != 0).sum(1)).float()
    return sum_log_degrees_after, new_n


def _updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]
    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]
    sum_log_degrees_after, new_n = _update_sum_log_degrees(
        sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    new_alpha = _compute_alpha(new_n, sum_log_degrees_after, d_min)
    new_ll = _compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after


def _likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency,
                              d_min, threshold=0.004):
    """
    Returns a mask of shape [N, N] (float, 0/1) where 1 = edge flip allowed.
    Filters out flips that would violate the log-likelihood degree constraint.
    """
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = _degree_sequence_log_likelihood(
        original_adjacency.sum(0), d_min)
    ll_current, _, _, _ = _degree_sequence_log_likelihood(
        modified_adjacency.sum(0), d_min)
    concat_deg = torch.cat((modified_adjacency.sum(0), original_adjacency.sum(0)))
    ll_comb, _, _, _ = _degree_sequence_log_likelihood(concat_deg, d_min)
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    new_lls, _, new_ns, new_sum_log_degrees = _updated_log_likelihood_for_edge_changes(
        node_pairs, modified_adjacency, d_min)

    n_combined = n_orig + new_ns
    new_sum_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = _compute_alpha(n_combined, new_sum_combined, d_min)
    new_ll_combined = _compute_log_likelihood(n_combined, alpha_combined, new_sum_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    allowed_edges = new_ratios < threshold
    try:
        mask_indices = node_pairs[allowed_edges.cpu().numpy().astype(bool)]
    except Exception:
        mask_indices = node_pairs[allowed_edges.numpy().astype(bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    if mask_indices.shape[0] > 0:
        allowed_mask[mask_indices.T] = 1
        allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


# ======================================================================
# Minimal GCN (for surrogate self-training)
# ======================================================================

class _GraphConvolution(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features)) if with_bias else None
        self._reset()

    def _reset(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = x @ self.weight
        if sp.issparse(x):
            support = torch.spmm(x, self.weight)
        out = torch.spmm(adj, support) if adj.is_sparse else adj @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class _GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True):
        super().__init__()
        self.gc1 = _GraphConvolution(nfeat, nhid, with_bias)
        self.gc2 = _GraphConvolution(nhid, nclass, with_bias)
        self.dropout = dropout
        self.with_relu = with_relu

    def initialize(self):
        self.gc1._reset()
        self.gc2._reset()

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        if self.with_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# ======================================================================
# BaseMeta
# ======================================================================

class BaseMeta(Module):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout,
                 train_iters, attack_features, lambda_, device,
                 with_bias=False, lr=0.01, with_relu=False):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.nfeat = nfeat
        self.nclass = nclass
        self.with_bias = with_bias
        self.with_relu = with_relu
        self.train_iters = train_iters
        self.attack_features = attack_features
        self.lambda_ = lambda_
        self.device = device
        self.nnodes = nnodes

        self.gcn = _GCN(nfeat=nfeat, nhid=hidden_sizes[0], nclass=nclass,
                        dropout=0.5, with_relu=False, with_bias=True)
        self.surrogate_optimizer = optim.Adam(
            self.gcn.parameters(), lr=lr, weight_decay=5e-4)

        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
        self.adj_changes.data.fill_(0)

    def filter_potential_singletons(self, modified_adj):
        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        return 1 - logical_and_symmetric

    def train_surrogate(self, features, adj, labels, idx_train, train_iters=200):
        print('  [MetaAttack] Training surrogate GCN for self-training labels...')
        surrogate = self.gcn.to(self.device)
        surrogate.initialize()

        adj_norm = _normalize_adj_tensor(adj, self.device)
        surrogate.train()
        for _ in range(train_iters):
            self.surrogate_optimizer.zero_grad()
            output = surrogate(features, adj_norm)
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss.backward()
            self.surrogate_optimizer.step()

        surrogate.eval()
        with torch.no_grad():
            output = surrogate(features, adj_norm)
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        surrogate.initialize()
        return labels_self_training

    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        t_d_min = torch.tensor(2.0, device=self.device)
        t_possible_edges = torch.tensor(
            np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T,
            dtype=torch.long, device=self.device)
        allowed_mask, current_ratio = _likelihood_ratio_filter(
            t_possible_edges, modified_adj, ori_adj, t_d_min, ll_cutoff)
        return allowed_mask, current_ratio


# ======================================================================
# Metattack (full second-order)
# ======================================================================

class Metattack(BaseMeta):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout,
                 train_iters, attack_features, device, lambda_=0.5,
                 with_relu=False, with_bias=False, lr=0.1, momentum=0.9):
        super().__init__(nfeat, hidden_sizes, nclass, nnodes, dropout,
                         train_iters, attack_features, lambda_, device,
                         with_bias=with_bias, with_relu=with_relu)
        self.momentum = momentum
        self.lr = lr

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        prev = nfeat
        for nhid in hidden_sizes:
            w = Parameter(torch.FloatTensor(prev, nhid).to(device))
            b = Parameter(torch.FloatTensor(nhid).to(device))
            self.weights.append(w)
            self.biases.append(b)
            self.w_velocities.append(torch.zeros_like(w))
            self.b_velocities.append(torch.zeros_like(b))
            prev = nhid

        w = Parameter(torch.FloatTensor(prev, nclass).to(device))
        b = Parameter(torch.FloatTensor(nclass).to(device))
        self.weights.append(w)
        self.biases.append(b)
        self.w_velocities.append(torch.zeros_like(w))
        self.b_velocities.append(torch.zeros_like(b))

        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1.0 / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def _forward_pass(self, features, adj_norm):
        h = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                h = adj_norm @ torch.spmm(h, w) + b
            else:
                h = adj_norm @ h @ w + b
            if self.with_relu:
                h = F.relu(h)
        return F.log_softmax(h, dim=1)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()
        for ix in range(len(self.weights)):
            self.weights[ix] = self.weights[ix].detach().requires_grad_(True)
            self.w_velocities[ix] = self.w_velocities[ix].detach().requires_grad_(True)
            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach().requires_grad_(True)
                self.b_velocities[ix] = self.b_velocities[ix].detach().requires_grad_(True)

        for _ in range(self.train_iters):
            output = self._forward_pass(features, adj_norm)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            weight_grads = torch.autograd.grad(
                loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [
                self.momentum * v + g
                for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(
                    loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [
                    self.momentum * v + g
                    for v, g in zip(self.b_velocities, bias_grads)]
            self.weights = [
                w - self.lr * v
                for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [
                    b - self.lr * v
                    for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled,
                      labels, labels_self_training):
        output = self._forward_pass(features, adj_norm)
        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(
            output[idx_unlabeled], labels_self_training[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = (self.lambda_ * loss_labeled
                           + (1 - self.lambda_) * loss_unlabeled)

        adj_grad = torch.autograd.grad(
            attack_loss, self.adj_changes, retain_graph=True)[0]
        return adj_grad

    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled,
                perturbations, ll_constraint=True, ll_cutoff=0.004):
        self.sparse_features = sp.issparse(features)
        labels_self_training = self.train_surrogate(
            features, ori_adj, labels, idx_train)

        for i in tqdm(range(perturbations), desc="  [MetaAttack] Perturbing"):
            adj_changes_sq = self.adj_changes - torch.diag(
                torch.diag(self.adj_changes, 0))
            adj_changes_symm = torch.clamp(
                adj_changes_sq + adj_changes_sq.t(), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            adj_norm = _normalize_adj_tensor(modified_adj, self.device)
            self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad = self.get_meta_grad(
                features, adj_norm, idx_train, idx_unlabeled,
                labels, labels_self_training)

            adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad * singleton_mask

            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(
                    modified_adj, ori_adj, ll_cutoff)
                allowed_mask = allowed_mask.to(self.device)
                adj_meta_grad = adj_meta_grad * allowed_mask

            adj_meta_argmax = torch.argmax(adj_meta_grad)
            row_idx, col_idx = _unravel_index(adj_meta_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (
                -2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (
                -2 * modified_adj[row_idx][col_idx] + 1)

        return self.adj_changes + ori_adj


# ======================================================================
# MetaApprox (faster approximation)
# ======================================================================

class MetaApprox(BaseMeta):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout,
                 train_iters, attack_features, lambda_, device,
                 with_relu=False, with_bias=False, lr=0.01):
        super().__init__(nfeat, hidden_sizes, nclass, nnodes, dropout,
                         train_iters, attack_features, lambda_, device,
                         with_bias=with_bias, with_relu=with_relu)
        self.lr = lr
        self.grad_sum = torch.zeros(nnodes, nnodes).to(device)

        self.weights = []
        self.biases = []
        prev = nfeat
        for nhid in hidden_sizes:
            self.weights.append(Parameter(torch.FloatTensor(prev, nhid).to(device)))
            self.biases.append(Parameter(torch.FloatTensor(nhid).to(device)))
            prev = nhid
        self.weights.append(Parameter(torch.FloatTensor(prev, nclass).to(device)))
        self.biases.append(Parameter(torch.FloatTensor(nclass).to(device)))
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1.0 / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)
        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

    def _forward_pass(self, features, adj_norm):
        h = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                h = adj_norm @ torch.spmm(h, w) + b
            else:
                h = adj_norm @ h @ w + b
            if self.with_relu:
                h = F.relu(h)
        return F.log_softmax(h, dim=1)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled,
                    labels, labels_self_training):
        adj_norm = _normalize_adj_tensor(modified_adj, self.device)
        for _ in range(self.train_iters):
            output = self._forward_pass(features, adj_norm)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(
                output[idx_unlabeled], labels_self_training[idx_unlabeled])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = (self.lambda_ * loss_labeled
                               + (1 - self.lambda_) * loss_unlabeled)

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)
            self.adj_changes.grad.zero_()
            self.grad_sum += torch.autograd.grad(
                attack_loss, self.adj_changes, retain_graph=True)[0]
            self.optimizer.step()

    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled,
                perturbations, ll_constraint=True, ll_cutoff=0.004):
        self.sparse_features = sp.issparse(features)
        labels_self_training = self.train_surrogate(
            features, ori_adj, labels, idx_train)

        for i in tqdm(range(perturbations), desc="  [MetaApprox] Perturbing"):
            adj_changes_sq = self.adj_changes - torch.diag(
                torch.diag(self.adj_changes, 0))
            adj_changes_symm = torch.clamp(
                adj_changes_sq + adj_changes_sq.t(), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            self._initialize()
            self.grad_sum.data.fill_(0)
            self.inner_train(features, modified_adj, idx_train, idx_unlabeled,
                             labels, labels_self_training)

            adj_meta_grad = self.grad_sum * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad * singleton_mask

            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(
                    modified_adj, ori_adj, ll_cutoff)
                allowed_mask = allowed_mask.to(self.device)
                adj_meta_grad = adj_meta_grad * allowed_mask

            adj_meta_approx_argmax = torch.argmax(adj_meta_grad)
            row_idx, col_idx = _unravel_index(
                adj_meta_approx_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (
                -2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (
                -2 * modified_adj[row_idx][col_idx] + 1)

        return self.adj_changes + ori_adj


# ======================================================================
# Main entry point: run attack on a PyG Data object
# ======================================================================

def run_metattack(data, n_perturbations, device='cpu', use_approx=False,
                  ll_constraint=False, hidden_sizes=None, train_iters=100,
                  lambda_=0, lr=0.1, momentum=0.9):
    """
    Run MetaAttack (Zügner & Günnemann, ICLR 2019) on a PyG Data object.

    Parameters
    ----------
    data            : torch_geometric.data.Data
    n_perturbations : int    — number of edge flips
    device          : str
    use_approx      : bool   — use MetaApprox (faster) instead of full Metattack
    ll_constraint   : bool   — enforce log-likelihood degree constraint
    hidden_sizes    : list   — hidden layer sizes for the meta-GCN (default [16])
    train_iters     : int    — inner training iterations per perturbation step
    lambda_         : float  — 0 = maximise loss on unlabeled, 1 = on labeled
    lr              : float  — inner SGD learning rate
    momentum        : float  — inner SGD momentum (Metattack only)

    Returns
    -------
    modified_adj : torch.Tensor  [N, N] dense, on CPU
    meta         : dict          diagnostics
    """
    if hidden_sizes is None:
        hidden_sizes = [16]

    # --- Convert PyG → dense tensors ---
    num_nodes = data.x.size(0)
    num_features = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1

    edge_index = data.edge_index.cpu().numpy()
    vals = np.ones(edge_index.shape[1], dtype=np.float32)
    adj_sp = sp.csr_matrix(
        (vals, (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes))
    adj_sp = adj_sp + adj_sp.T
    adj_sp[adj_sp > 1] = 1
    adj_sp.eliminate_zeros()

    features_np = data.x.cpu().numpy().astype(np.float32)
    labels_np   = data.y.cpu().numpy()

    idx_train = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    idx_val   = data.val_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    idx_test  = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    idx_unlabeled = np.union1d(idx_val, idx_test)

    # Move to device as dense tensors
    adj      = torch.FloatTensor(adj_sp.toarray()).to(device)
    features = torch.FloatTensor(features_np).to(device)
    labels   = torch.LongTensor(labels_np).to(device)
    idx_train     = torch.LongTensor(idx_train).to(device)
    idx_unlabeled = torch.LongTensor(idx_unlabeled).to(device)

    print(f"  [MetaAttack] Graph: {num_nodes} nodes, {int(adj.sum().item()//2)} edges, "
          f"{num_features} feats, {num_classes} classes")
    print(f"  [MetaAttack] Train: {len(idx_train)}, Unlabeled: {len(idx_unlabeled)}, "
          f"perturbations: {n_perturbations}, approx={use_approx}")

    if use_approx:
        attacker = MetaApprox(
            nfeat=num_features,
            hidden_sizes=hidden_sizes,
            nclass=num_classes,
            nnodes=num_nodes,
            dropout=0.5,
            train_iters=train_iters,
            attack_features=False,
            lambda_=lambda_,
            device=device,
            with_relu=False,
            with_bias=False,
            lr=lr,
        ).to(device)
    else:
        attacker = Metattack(
            nfeat=num_features,
            hidden_sizes=hidden_sizes,
            nclass=num_classes,
            nnodes=num_nodes,
            dropout=0.5,
            train_iters=train_iters,
            attack_features=False,
            device=device,
            lambda_=lambda_,
            with_relu=False,
            with_bias=False,
            lr=lr,
            momentum=momentum,
        ).to(device)

    modified_adj = attacker(
        features, adj, labels,
        idx_train, idx_unlabeled,
        n_perturbations,
        ll_constraint=ll_constraint,
    )

    # Diagnostics
    diff = (modified_adj - adj).abs()
    n_changed = int((diff > 0.5).sum().item())
    print(f"  [MetaAttack] Adjacency entries changed: {n_changed} "
          f"(expected ~{2 * n_perturbations} for symmetric adj)")

    return modified_adj.detach().cpu(), {
        "n_perturbations": n_perturbations,
        "entries_changed": n_changed,
        "method": "MetaApprox" if use_approx else "Metattack",
    }

# -*- coding: utf-8 -*-

import torch
import os


def evaluate(out, labels):
    """Calculates accuracy between prediction and ground truth."""
    preds = out.argmax(dim=1)
    correct = preds == labels
    if correct.size(0) == 0:
        return 0.0
    acc = int(correct.sum()) / int(correct.size(0))
    return acc


def store_checkpoint(paper, dataset, model, train_acc, val_acc, test_acc, epoch=-1):
    save_dir = f"./checkpoints/{paper}/{dataset}"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, "best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))


def store_vote_checkpoint(paper, dataset, models, train_acc, val_acc, test_acc, epoch=-1):
    save_dir = f"./checkpoints/{paper}/{dataset}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if epoch == -1:
        for i in range(len(models)):
            checkpoint = {'model_state_dict': models[i].state_dict(),
                          'train_acc': train_acc,
                          'val_acc': val_acc,
                          'test_acc': test_acc}
            torch.save(checkpoint, os.path.join(save_dir, f"best_model_{i}"))
    else:
        for i in range(len(models)):
            checkpoint = {'model_state_dict': models[i].state_dict(),
                          'train_acc': train_acc,
                          'val_acc': val_acc,
                          'test_acc': test_acc}
            torch.save(checkpoint, os.path.join(save_dir, f"best_model_{epoch}_{i}"))


def store_secert_checkpoint(paper, dataset, model_dict, train_acc, val_acc, test_acc):
    """Store SECert model (shared backbone + heads) as a single checkpoint."""
    save_dir = f"./checkpoints/{paper}/{dataset}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        'model_state_dict': model_dict,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc
    }
    torch.save(checkpoint, os.path.join(save_dir, "best_model"))


def load_best_model(best_epoch, paper, dataset, model, eval_enabled):
    if best_epoch == -1:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_model")
    else:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    if eval_enabled:
        model.eval()
    return model


def train_model(model, edge_index, x, labels, train_mask, val_mask, test_mask, train_args):
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(0, train_args["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_args["clip_max"])
        optimizer.step()
        with torch.no_grad():
            out = model(x, edge_index)
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])
        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
        if val_acc > best_val_acc:
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(train_args["paper"], train_args["dataset"], model, train_acc, val_acc, test_acc)
        if epoch - best_epoch > train_args["early_stopping"] and best_val_acc > 0.99:
            break

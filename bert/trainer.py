import torch
import torch.nn as nn
from typing import Callable

from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        n_epochs: int,
        validate_every: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn,
        optimizer,
        metrics: dict[str, Callable],
        log_path: str | None = "bert/logs/bert_hist.csv",
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.validate_every = validate_every
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.log_path = log_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # history: [{epoch: 0, split: train, loss: 10, etc.}, {}]
        self.history = []
        metrics_str = ""
        for metric in metrics.keys():
            metrics_str += f",pos_{metric},neg_{metric}"

        if self.log_path is not None:
            with open(self.log_path, "w") as f:
                f.write(f"epoch,step,split,loss{metrics_str}")

        self.curr_epoch = -1
        self.curr_step = -1

    def fit(self):
        for i in tqdm(range(self.n_epochs)):
            self.curr_epoch = i
            self.train()
            if i % self.validate_every == 0:
                self.validate()

    def train(self):
        for i, (tag_embs, embs, neg_embs) in enumerate(self.train_loader):
            self.curr_step = i
            tag_embs = {k: v.to(self.device) for k, v in tag_embs.items()}
            history = self.train_step(tag_embs, embs, neg_embs)
            self.history.append(history)
            self.log_step(history)

    def train_step(self, x, pos, neg):
        history = {"epoch": self.curr_epoch, "step": self.curr_step, "split": "train"}
        self.model.train()
        anchor = self.model.forward(x)
        loss = self.loss_fn(anchor, pos, neg)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        history["loss"] = loss.item()

        pos_metrics = self.get_metrics(pos, anchor, True)
        neg_metrics = self.get_metrics(neg, anchor, False)
        for (pos_k, pos_v), (neg_k, neg_v) in zip(
            pos_metrics.items(), neg_metrics.items()
        ):
            history[pos_k] = pos_v
            history[neg_k] = neg_v

        return history

    def get_metrics(self, y_true, y_pred, is_pos: bool) -> dict[str, float]:
        """outputs dict[str, value]"""
        if is_pos:
            metric_str = "pos_"
        else:
            metric_str = "neg_"
        metrics = {}
        for metric, metric_fn in self.metrics.items():
            result = metric_fn(y_true, y_pred)
            metrics[metric_str + metric] = result
        return metrics

    def validate(self):
        with torch.no_grad():
            histories = []
            for tag_embs, embs, neg_embs in self.val_loader:
                tag_embs = {k: v.to(self.device) for k, v in tag_embs.items()}
                history = self.validation_step(tag_embs, embs, neg_embs)
                histories.append(history)

        total_loss = 0
        total_pos_cos = 0
        total_pos_roc_aur = 0
        total_neg_cos = 0
        total_neg_roc_aur = 0
        for history in histories:
            total_loss += history["loss"]
            total_pos_cos += history["pos_cosine_similarity"]
            total_pos_roc_aur += history["pos_roc_score"]
            total_neg_cos += history["neg_cosine_similarity"]
            total_neg_roc_aur += history["neg_roc_score"]
        total_loss /= len(histories)
        total_pos_cos /= len(histories)
        total_pos_roc_aur /= len(histories)
        total_neg_cos /= len(histories)
        total_neg_roc_aur /= len(histories)
        history = {
            "epoch": self.curr_epoch,
            "step": -1,
            "split": "val",
            "pos_cosine_similarity": total_pos_cos,
            "neg_cosine_similarity": total_neg_cos,
            "pos_roc_score": total_pos_roc_aur,
            "neg_roc_score": total_neg_roc_aur,
        }
        self.log_step(history)

    def validation_step(self, x, pos, neg):
        history = {"epoch": self.curr_epoch, "step": -1, "split": "train"}

        self.model.eval()
        anchor = self.model.forward(x)
        loss = self.loss_fn(anchor, pos, neg)
        history["loss"] = loss.item()

        pos_metrics = self.get_metrics(pos, anchor, True)
        neg_metrics = self.get_metrics(neg, anchor, False)
        for (pos_k, pos_v), (neg_k, neg_v) in zip(
            pos_metrics.items(), neg_metrics.items()
        ):
            history[pos_k] = pos_v
            history[neg_k] = neg_v

        return history

    def log_step(self, history: dict, verbose: bool = False):
        if self.log_path is None:
            return

        epoch = history["epoch"]
        step = history["step"]
        split = history["split"]
        loss = history["loss"]
        pos_cos = history["pos_cosine_similarity"]
        pos_roc_aur = history["pos_roc_score"]
        neg_cos = history["neg_cosine_similarity"]
        neg_roc_aur = history["neg_roc_score"]

        with open(self.log_path, "a") as f:
            f.write(
                f"{epoch},{step},{split},{loss},{pos_cos},{neg_cos},{pos_roc_aur},{neg_roc_aur}"
            )
        if verbose:
            epoch = self.curr_epoch
            print(
                f"Epoch {epoch}: train loss: {loss}, cos_sim: {pos_cos}, roc_aur: {pos_roc_aur}"
            )

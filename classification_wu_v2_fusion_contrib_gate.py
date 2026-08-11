import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from tqdm.auto import tqdm

from sklearn.metrics import f1_score, roc_auc_score

from layers.multireslayer import MultiresLayer
from utils import (
    count_parameters,
    apply_norm,
    ddp_setup,
    get_cosine_schedule_with_warmup
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# =========================
# Dataset
# =========================

def estimate_effective_lengths(x, eps_ratio=0.02, min_length=64):
    """
    x: torch.Tensor, shape (N, C, L)
    根据幅值阈值估计每条次声的有效长度。
    思路：对通道取平均能量，找到最后一个超过阈值的位置。

    eps_ratio:
        阈值 = 每条样本最大能量 * eps_ratio
    """
    with torch.no_grad():
        energy = x.abs().mean(dim=1)  # (N, L)
        max_energy = energy.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        threshold = max_energy * eps_ratio
        valid = energy > threshold

        lengths = []
        for i in range(valid.shape[0]):
            idx = torch.where(valid[i])[0]
            if len(idx) == 0:
                lengths.append(min_length)
            else:
                lengths.append(max(int(idx[-1].item()) + 1, min_length))

        lengths = torch.tensor(lengths, dtype=torch.long)
    return lengths


class SeismoAcousticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_seismic,
        X_infra,
        y,
        infra_downsample=11,
        infra_eps_ratio=0.02,
        seismic_max_length=3600,
        infra_max_length=None,
    ):
        """
        X_seismic: (N, 3, 3600, 3)  -> (N, 9, 3600)
        X_infra:   (N, 3, 43601, 1) -> (N, 3, 43601) -> downsample
        y:         (N,) or (N, 1)
        """

        Xs = torch.from_numpy(X_seismic).float()
        Xi = torch.from_numpy(X_infra).float()

        # seismic: (N, station, length, component)
        # -> (N, station, component, length)
        # -> (N, station * component, length)
        Xs = Xs.permute(0, 1, 3, 2)
        Xs = Xs.reshape(Xs.shape[0], Xs.shape[1] * Xs.shape[2], Xs.shape[3])

        # infra: (N, station, length, 1)
        # -> (N, station, 1, length)
        # -> (N, station, length)
        Xi = Xi.permute(0, 1, 3, 2)
        Xi = Xi.reshape(Xi.shape[0], Xi.shape[1] * Xi.shape[2], Xi.shape[3])

        if infra_downsample > 1:
            Xi = Xi[..., ::infra_downsample]

        self.seismic = Xs
        self.infra = Xi

        self.seismic_lengths = torch.full(
            (Xs.shape[0],), min(seismic_max_length, Xs.shape[-1]), dtype=torch.long
        )

        self.infra_lengths = estimate_effective_lengths(
            Xi,
            eps_ratio=infra_eps_ratio,
            min_length=64,
        )

        if infra_max_length is not None:
            self.infra_lengths = self.infra_lengths.clamp(max=infra_max_length)

        y = torch.from_numpy(y).float()
        if y.ndim > 1:
            y = y[:, 0]
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "seismic": self.seismic[idx],
            "infra": self.infra[idx],
            "seismic_length": self.seismic_lengths[idx],
            "infra_length": self.infra_lengths[idx],
            "label": self.labels[idx].unsqueeze(0),
        }


# =========================
# Model
# =========================

def masked_meanpool(x, lengths):
    """
    x: (B, D, L)
    lengths: (B,)
    """
    B, D, L = x.shape
    lengths = lengths.to(x.device).clamp(min=1, max=L)
    mask = torch.arange(L, device=x.device)[None, :] < lengths[:, None]
    mask = mask[:, None, :].float()
    return (x * mask).sum(dim=-1) / lengths[:, None].float()


class MultiresEncoder(nn.Module):
    def __init__(
        self,
        d_input,
        d_model=128,
        n_layers=4,
        dropout=0.1,
        batchnorm=False,
        max_length=None,
        hinit=None,
        depth=None,
        tree_select="fading",
        d_mem=None,
        kernel_size=2,
        indep_res_init=False,
    ):
        super().__init__()

        self.batchnorm = batchnorm
        self.max_length = max_length

        self.encoder = nn.Conv1d(d_input, d_model, 1)
        self.seq_layers = nn.ModuleList()
        self.mixing_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        norm_func = nn.BatchNorm1d if batchnorm else nn.LayerNorm

        for _ in range(n_layers):
            self.seq_layers.append(
                MultiresLayer(
                    d_model,
                    kernel_size=kernel_size,
                    depth=depth,
                    wavelet_init=hinit,
                    tree_select=tree_select,
                    seq_len=max_length,
                    dropout=dropout,
                    memory_size=d_mem,
                    indep_res_init=indep_res_init,
                )
            )

            self.mixing_layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model, 2 * d_model, 1),
                    nn.GLU(dim=-2),
                    nn.Dropout1d(dropout),
                )
            )

            self.norms.append(norm_func(d_model))

    def forward(self, x, lengths=None):
        """
        x: (B, C, L)
        lengths: (B,) or None
        """
        if self.max_length is not None:
            num_pad = self.max_length - x.shape[-1]
            if num_pad > 0:
                x = nn.functional.pad(x, (0, num_pad), "constant", 0)
            elif num_pad < 0:
                x = x[..., :self.max_length]
                if lengths is not None:
                    lengths = lengths.clamp(max=self.max_length)

        x = self.encoder(x)

        for layer, mixing_layer, norm in zip(
            self.seq_layers, self.mixing_layers, self.norms
        ):
            x_orig = x
            x = layer(x)
            x = mixing_layer(x)
            x = x + x_orig
            x = apply_norm(x, norm, self.batchnorm)

        if lengths is not None:
            x = masked_meanpool(x, lengths)
        else:
            x = x.mean(dim=-1)

        return x


class InfraGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_infra):
        return self.net(z_infra)


class PrototypeLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, h, y):
        """
        h: (B, D)
        y: (B, 1)
        """
        y = y.view(-1).long()
        h = nn.functional.normalize(h, dim=1)

        if (y == 0).sum() == 0 or (y == 1).sum() == 0:
            return h.new_tensor(0.0)

        c0 = h[y == 0].mean(dim=0)
        c1 = h[y == 1].mean(dim=0)

        c0 = nn.functional.normalize(c0, dim=0)
        c1 = nn.functional.normalize(c1, dim=0)

        d0 = 1.0 - torch.matmul(h, c0)
        d1 = 1.0 - torch.matmul(h, c1)

        d_pos = torch.where(y == 0, d0, d1)
        d_neg = torch.where(y == 0, d1, d0)

        return torch.relu(self.margin + d_pos - d_neg).mean()


class SeismoAcousticFusionNet(nn.Module):
    def __init__(
        self,
        seismic_input=9,
        infra_input=3,
        d_model=128,
        n_layers=4,
        dropout=0.1,
        batchnorm=False,
        seismic_max_length=3600,
        infra_max_length=3964,
        hinit=None,
        depth=None,
        tree_select="fading",
        d_mem=None,
        kernel_size=2,
        indep_res_init=False,
    ):
        super().__init__()

        self.seismic_encoder = MultiresEncoder(
            d_input=seismic_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            batchnorm=batchnorm,
            max_length=seismic_max_length,
            hinit=hinit,
            depth=depth,
            tree_select=tree_select,
            d_mem=d_mem,
            kernel_size=kernel_size,
            indep_res_init=indep_res_init,
        )

        self.infra_encoder = MultiresEncoder(
            d_input=infra_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            batchnorm=batchnorm,
            max_length=infra_max_length,
            hinit=hinit,
            depth=depth,
            tree_select=tree_select,
            d_mem=d_mem,
            kernel_size=kernel_size,
            indep_res_init=indep_res_init,
        )

        self.proj_seismic = nn.Linear(d_model, d_model)
        self.proj_infra = nn.Linear(d_model, d_model)

        self.infra_gate = InfraGate(d_model)

        # Fusion classifier: predicts from the fused representation h.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Seismic-only auxiliary classifier.
        # It provides a seismic reference loss for the contribution-aware gate loss.
        self.seismic_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, seismic, infra, seismic_lengths=None, infra_lengths=None):
        z_s = self.seismic_encoder(seismic, seismic_lengths)
        z_i = self.infra_encoder(infra, infra_lengths)

        z_s = self.proj_seismic(z_s)
        z_i = self.proj_infra(z_i)

        g_i = self.infra_gate(z_i)

        # 地震恒有效，次声由 gate 控制贡献
        h = z_s + g_i * z_i

        logit_fusion = self.classifier(h)
        logit_seismic = self.seismic_classifier(z_s)

        return logit_fusion, logit_seismic, h, g_i


# =========================
# Optimizer
# =========================

def setup_optimizer(model, lr, weight_decay, epochs, iters_per_epoch, warmup):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * iters_per_epoch

    if warmup > 0:
        warmup_steps = warmup * iters_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            total_steps,
        )

    return optimizer, scheduler


# =========================
# Train / Eval
# =========================

def train_one_epoch(
    device,
    epoch,
    trainloader,
    model,
    optimizer,
    scheduler,
    bce_criterion,
    proto_criterion,
    lambda_proto=0.1,
    beta_gate=0.001,
    alpha=0.2,
):
    model.train()
    trainloader.sampler.set_epoch(epoch)

    total_loss = 0.0
    total_samples = 0

    true_targets = []
    pred_targets = []
    gate_values = []

    pbar = enumerate(trainloader)
    if device == 0:
        pbar = tqdm(pbar, total=len(trainloader))

    for batch_idx, batch in pbar:
        seismic = batch["seismic"].to(device)
        infra = batch["infra"].to(device)
        seismic_lengths = batch["seismic_length"].to(device)
        infra_lengths = batch["infra_length"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs, outputs_seismic, h, g_i = model(
            seismic,
            infra,
            seismic_lengths=seismic_lengths,
            infra_lengths=infra_lengths,
        )

        # Fusion classification loss.
        loss_bce = bce_criterion(outputs, targets)

        # Seismic-only reference loss.
        # This branch is treated as the always-available baseline.
        loss_bce_seismic = bce_criterion(outputs_seismic, targets)

        # Prototype clustering loss on the fused representation.
        loss_proto = proto_criterion(h, targets)

        # Contribution-aware gate loss.
        # If fusion is worse than seismic-only, the model is penalized and
        # the gate learns to suppress harmful infrasound contribution.
        # detach() keeps the seismic-only branch as a stable reference.
        loss_aux = alpha * loss_bce_seismic
        loss_gate = torch.relu(loss_bce - loss_bce_seismic.detach())

        loss = loss_bce + loss_aux + lambda_proto * loss_proto + beta_gate * loss_gate

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

        true_targets.append(targets.detach().cpu().numpy())
        pred_targets.append(outputs.detach().cpu().numpy())
        gate_values.append(g_i.detach().cpu().numpy())

        if device == 0:
            avg_loss = total_loss / total_samples
            pbar.set_description(
                f"Epoch {epoch} | Loss {avg_loss:.4f} | "
                f"BCE {loss_bce.item():.4f} | "
                f"BCE-S {loss_bce_seismic.item():.4f} | "
                f"Aux {loss_aux.item():.4f} | "
                f"Proto {loss_proto.item():.4f} | "
                f"GateLoss {loss_gate.item():.4f} | "
                f"GateMean {g_i.mean().item():.3f}"
            )

    true_targets = np.concatenate(true_targets).flatten()
    pred_targets = np.concatenate(pred_targets).flatten()
    gate_values = np.concatenate(gate_values).flatten()

    probs = 1 / (1 + np.exp(-pred_targets))
    preds = (probs > 0.5).astype(int)
    labels = true_targets.astype(int)

    acc = np.mean(preds == labels)
    f1 = f1_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return total_loss / total_samples, acc, f1, auc, float(gate_values.mean())


@torch.no_grad()
def evaluate(
    device,
    dataloader,
    model,
    bce_criterion,
    proto_criterion=None,
    lambda_proto=0.1,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    true_targets = []
    pred_targets = []
    gate_values = []

    pbar = enumerate(dataloader)
    if device == 0:
        pbar = tqdm(pbar, total=len(dataloader))

    for batch_idx, batch in pbar:
        seismic = batch["seismic"].to(device)
        infra = batch["infra"].to(device)
        seismic_lengths = batch["seismic_length"].to(device)
        infra_lengths = batch["infra_length"].to(device)
        targets = batch["label"].to(device)

        outputs, outputs_seismic, h, g_i = model(
            seismic,
            infra,
            seismic_lengths=seismic_lengths,
            infra_lengths=infra_lengths,
        )

        loss = bce_criterion(outputs, targets)
        if proto_criterion is not None:
            loss = loss + lambda_proto * proto_criterion(h, targets)

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

        true_targets.append(targets.cpu().numpy())
        pred_targets.append(outputs.cpu().numpy())
        gate_values.append(g_i.cpu().numpy())

    true_targets = np.concatenate(true_targets).flatten()
    pred_targets = np.concatenate(pred_targets).flatten()
    gate_values = np.concatenate(gate_values).flatten()

    probs = 1 / (1 + np.exp(-pred_targets))
    preds = (probs > 0.5).astype(int)
    labels = true_targets.astype(int)

    acc = np.mean(preds == labels)
    f1 = f1_score(labels, preds)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return total_loss / total_samples, acc, f1, auc, float(gate_values.mean())


# =========================
# Main
# =========================

def main(rank, world_size, args):
    ddp_setup(rank, world_size, args.port)

    assert args.batch_size % world_size == 0
    per_device_batch_size = args.batch_size // world_size

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split = args.split

    # X_train_seismic = np.load(
    #     f"{args.data_root}/Train_Data/6x/Split_{split}/X_seismic1.npy"
    # )
    # X_train_infra = np.load(
    #     f"{args.data_root}/Train_Data/6x/Split_{split}/X_infra1.npy"
    # )
    # y_train = np.load(
    #     f"{args.data_root}/Train_Data/6x/Split_{split}/y.npy"
    # )

    # X_test_seismic = np.load(
    #     f"{args.data_root}/Test_Data/Split_{split}/X_seismic1.npy"
    # )
    # X_test_infra = np.load(
    #     f"{args.data_root}/Test_Data/Split_{split}/X_infra1.npy"
    # )
    # y_test = np.load(
    #     f"{args.data_root}/Test_Data/Split_{split}/y.npy"
    # )

    ### data2transfer
    X_train_seismic = np.load(
        f"{args.data_root}/Train90/X_seismic1.npy"
    )
    X_train_infra = np.load(
        f"{args.data_root}/Train90/X_infra1.npy"
    )
    y_train = np.load(
        f"{args.data_root}/Train90/y.npy"
    )

    X_test_seismic = np.load(
        f"{args.data_root}/Val10/X_seismic1.npy"
    )
    X_test_infra = np.load(
        f"{args.data_root}/Val10/X_infra1.npy"
    )
    y_test = np.load(
        f"{args.data_root}/Val10/y.npy"
    )


    if y_train.ndim > 1:
        y_train = y_train[:, 0]
    if y_test.ndim > 1:
        y_test = y_test[:, 0]

    trainset = SeismoAcousticDataset(
        X_train_seismic,
        X_train_infra,
        y_train,
        infra_downsample=args.infra_downsample,
        infra_eps_ratio=args.infra_eps_ratio,
        seismic_max_length=args.seismic_max_length,
        infra_max_length=args.infra_max_length,
    )

    testset = SeismoAcousticDataset(
        X_test_seismic,
        X_test_infra,
        y_test,
        infra_downsample=args.infra_downsample,
        infra_eps_ratio=args.infra_eps_ratio,
        seismic_max_length=args.seismic_max_length,
        infra_max_length=args.infra_max_length,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(trainset),
        persistent_workers=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    torch.cuda.set_device(rank)

    model = SeismoAcousticFusionNet(
        seismic_input=9,
        infra_input=3,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
        seismic_max_length=args.seismic_max_length,
        infra_max_length=args.infra_max_length,
        hinit=args.hinit,
        depth=args.depth,
        tree_select=args.tree_select,
        d_mem=args.d_mem,
        kernel_size=args.kernel_size,
        indep_res_init=args.indep_res_init,
    ).to(rank)

    if args.batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    bce_criterion = nn.BCEWithLogitsLoss()
    proto_criterion = PrototypeLoss(margin=args.proto_margin)

    optimizer, scheduler = setup_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        iters_per_epoch=len(trainloader),
        warmup=args.warmup,
    )

    if args.resume is not None:
        log_dir = args.resume
        map_location = {"cuda:0": f"cuda:{rank}"}
        checkpoint = torch.load(
            os.path.join(log_dir, "best_ckpt.pth"),
            map_location=map_location,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        log_dir = os.path.join(
            args.log_root,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
        start_epoch = 0

    if rank == 0:
        logger = logging.getLogger("fusion")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        console_handler = logging.StreamHandler()

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"Total parameters: {count_parameters(model)}")
        logger.info(f"Train seismic shape: {X_train_seismic.shape}")
        logger.info(f"Train infra shape: {X_train_infra.shape}")
        logger.info(f"Test seismic shape: {X_test_seismic.shape}")
        logger.info(f"Test infra shape: {X_test_infra.shape}")

        with open(os.path.join(log_dir, "args.txt"), "w") as f:
            f.write("\n".join(sys.argv[1:]))

        best_test_f1 = 0.0
        patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_f1, train_auc, train_gate = train_one_epoch(
            rank,
            epoch,
            trainloader,
            model,
            optimizer,
            scheduler,
            bce_criterion,
            proto_criterion,
            lambda_proto=args.lambda_proto,
            beta_gate=args.beta_gate,
            alpha=args.alpha,
        )

        should_stop = torch.tensor(0, device=rank)

        if rank == 0:
            test_loss, test_acc, test_f1, test_auc, test_gate = evaluate(
                rank,
                testloader,
                model,
                bce_criterion,
                proto_criterion=proto_criterion,
                lambda_proto=args.lambda_proto,
            )

            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} "
                f"F1 {train_f1:.4f} AUC {train_auc:.4f} Gate {train_gate:.4f} | "
                f"Test Loss {test_loss:.4f} Acc {test_acc:.4f} "
                f"F1 {test_f1:.4f} AUC {test_auc:.4f} Gate {test_gate:.4f}"
            )

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                patience_counter = 0

                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "test_f1": test_f1,
                    "test_auc": test_auc,
                    "test_gate": test_gate,
                }

                torch.save(state, os.path.join(log_dir, "best_ckpt.pth"))
            else:
                patience_counter += 1
                logger.info(f"EarlyStopping counter: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. Best Test F1: {best_test_f1:.4f}"
                )
                should_stop.fill_(1)

        dist.broadcast(should_stop, src=0)

        if should_stop.item() == 1:
            break

    if rank == 0:
        logger.info(f"FINAL: best test f1={best_test_f1:.4f}")

    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--data_root",
        default="/share/org/ZJUZHUZL/zju_wuhf/DL_Seismoacoustic_Fusion-main/DATA/Data2transfer",
        type=str,
    )
    parser.add_argument("--split", default=2, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=64, type=int)

    parser.add_argument("--seismic_max_length", default=3600, type=int)
    parser.add_argument("--infra_downsample", default=11, type=int)
    parser.add_argument("--infra_max_length", default=3964, type=int)
    parser.add_argument("--infra_eps_ratio", default=0.02, type=float)

    # Training
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--patience", default=40, type=int)

    # Loss
    parser.add_argument("--lambda_proto", default=0.1, type=float)
    parser.add_argument("--proto_margin", default=0.5, type=float)
    parser.add_argument("--beta_gate", default=0.1, type=float)
    parser.add_argument("--alpha", default=0.2, type=float)

    # Model
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--hinit", default=None, type=str)
    parser.add_argument("--depth", default=None, type=int)
    parser.add_argument("--tree_select", default="fading", choices=["uniform", "fading"])
    parser.add_argument("--d_mem", default=None, type=int)
    parser.add_argument("--kernel_size", default=2, type=int)
    parser.add_argument("--indep_res_init", action="store_true")

    # Others
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--log_root", default="logs/transfer_retrain_SA", type=str)
    parser.add_argument("--port", default="12669", type=str)
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
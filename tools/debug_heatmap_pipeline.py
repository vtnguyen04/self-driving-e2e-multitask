
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from pathlib import Path
from neuro_pilot.utils.tqdm import TQDM as tqdm
import logging
from dataclasses import dataclass, field
from typing import Optional

from neuro_pilot.cfg.schema import load_config
from neuro_pilot.data.neuro_pilot_dataset_v2 import NeuroPilotDataset, custom_collate_fn
from neuro_pilot.data.augment import StandardAugmentor
from neuro_pilot.data.build import build_dataloader
from neuro_pilot.nn.tasks import DetectionModel
from neuro_pilot.utils.losses import CombinedLoss, HeatmapLoss
from neuro_pilot.utils.ops import non_max_suppression, xywh2xyxy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    dataset_yaml: str = "/home/quynhthu/Documents/AI-project/e2e/tools/labeler/data/exports/project_12/v1/data.yaml"
    batch_size: int = 16
    image_size: int = 320
    num_workers: int = 1
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    nc: int = 14
    conf_thres: float = 0.25
    num_vis_samples: int = 16
    val_every_n_epochs: int = 1
    save_dir: str = "runs/train"
    lambda_heatmap: float = 10.0
    lambda_traj: float = 7.5
    lambda_det: float = 2.5
    lambda_cls: float = 1.0


@dataclass
class EpochMetrics:
    total: float = 0.0
    box: float = 0.0
    heatmap: float = 0.0
    cls: float = 0.0
    traj: float = 0.0
    count: int = 0

    def update(self, loss_dict: dict) -> None:
        self.total += loss_dict["total"].item()
        self.box += loss_dict.get("box", 0.0).item() if torch.is_tensor(loss_dict.get("box")) else loss_dict.get("box", 0.0)
        self.heatmap += loss_dict.get("heatmap", 0.0).item() if torch.is_tensor(loss_dict.get("heatmap")) else loss_dict.get("heatmap", 0.0)
        self.cls += loss_dict.get("cls", 0.0).item() if torch.is_tensor(loss_dict.get("cls")) else loss_dict.get("cls", 0.0)
        self.traj += loss_dict.get("traj", 0.0).item() if torch.is_tensor(loss_dict.get("traj")) else loss_dict.get("traj", 0.0)
        self.count += 1

    def averages(self) -> dict:
        if self.count == 0:
            return {}
        return {
            "total": self.total / self.count,
            "box": self.box / self.count,
            "heatmap": self.heatmap / self.count,
            "cls": self.cls / self.count,
            "traj": self.traj / self.count,
        }


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    img = (img_tensor * std + mean).permute(1, 2, 0).cpu().numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def draw_gt_on_image(
    img: np.ndarray,
    batch: dict,
    idx: int,
    image_size: int,
) -> np.ndarray:
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    S = image_size

    gt_mask = batch["batch_idx"] == idx
    gt_boxes = batch["bboxes"][gt_mask]
    for box in gt_boxes:
        x1, y1, x2, y2 = (
            xywh2xyxy(box.cpu().numpy().reshape(1, 4)).flatten() * S
        )
        cv2.rectangle(
            canvas,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )

    if batch.get("waypoints_mask", torch.ones(batch["image"].shape[0]))[idx] > 0:
        for wp in batch["waypoints"][idx]:
            # Skip padding (0,0) or check if they are real
            if wp.sum() == 0: continue
            x = int((wp[0].item() + 1) / 2 * S)
            y = int((wp[1].item() + 1) / 2 * S)
            cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

    return canvas


def draw_pred_on_image(
    img: np.ndarray,
    preds: dict,
    detections: list,
    idx: int,
    image_size: int,
) -> np.ndarray:
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    S = image_size

    if idx < len(detections) and detections[idx].numel() > 0:
        for det in detections[idx]:
            x1, y1, x2, y2 = det[:4].cpu().numpy()
            cv2.rectangle(
                canvas,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )

    if "waypoints" in preds:
        for wp in preds["waypoints"][idx]:
            x = int((wp[0].item() + 1) / 2 * S)
            y = int((wp[1].item() + 1) / 2 * S)
            cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

    return canvas


def render_heatmap(hm: np.ndarray, image_size: int) -> np.ndarray:
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
    hm_color = cv2.applyColorMap(
        (hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.resize(hm_color, (image_size, image_size))


class Visualizer:
    def __init__(self, heatmap_loss: HeatmapLoss, image_size: int = 320):
        self.heatmap_loss = heatmap_loss
        self.image_size = image_size

    def create_comparison_grid(
        self,
        idx: int,
        batch: dict,
        preds: dict,
        detections: list,
    ) -> np.ndarray:
        S = self.image_size
        orig_img = denormalize(batch["image"][idx])

        img_gt = draw_gt_on_image(orig_img, batch, idx, S)
        img_pred = draw_pred_on_image(orig_img, preds, detections, idx, S)

        H, W = preds["heatmap"].shape[2:]
        has_wp = batch.get("waypoints_mask", torch.ones(batch["image"].shape[0]))[idx] > 0

        if has_wp:
            gt_hm = self.heatmap_loss.generate_heatmap(
                batch["waypoints"][idx].unsqueeze(0), H, W
            ).cpu().numpy()[0, 0]
        else:
            gt_hm = np.zeros((H, W))

        pred_hm = torch.sigmoid(preds["heatmap"][idx, 0]).detach().cpu().numpy()

        top = np.hstack((img_gt, img_pred))
        bottom = np.hstack((render_heatmap(gt_hm, S), render_heatmap(pred_hm, S)))
        return np.vstack((top, bottom))


def build_model_and_optimizer(cfg: TrainConfig, device: torch.device):
    repo_root = Path(__file__).resolve().parent.parent
    model_cfg = repo_root / "neuro_pilot/cfg/models/yolo_style.yaml"
    model = DetectionModel(
        cfg=str(model_cfg),
        nc=cfg.nc,
        scale='s',
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * 0.01,
    )

    return model, optimizer, scheduler


def build_criterion(cfg: TrainConfig, model, device: torch.device) -> CombinedLoss:
    base_config = load_config()
    base_config.data.image_size = cfg.image_size

    criterion = CombinedLoss(base_config, model, device=device)
    criterion.lambda_heatmap = cfg.lambda_heatmap
    criterion.lambda_traj = cfg.lambda_traj
    criterion.lambda_det = cfg.lambda_det
    criterion.lambda_cls = cfg.lambda_cls
    return criterion


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    metrics: dict,
    cfg=None,
) -> None:
    # Standardize names for inference
    names = getattr(model, "names", {i: f"class_{i}" for i in range(14)})
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "model_cfg": "neuro_pilot/cfg/models/yolo_style.yaml",
            "names": names,
            "cfg": cfg,
            "date": __import__("datetime").datetime.now().isoformat(),
        },
        path,
    )


def run_validation(
    model: nn.Module,
    val_loader,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    model.eval()
    metrics = EpochMetrics()

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            targets = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = model(imgs, cmd_onehot=targets["command"])
            loss_dict = criterion.advanced(preds, targets)

            if not torch.isnan(loss_dict["total"]):
                metrics.update(loss_dict)

    model.train()
    return metrics.averages()


def save_visualizations(
    model: nn.Module,
    val_loader,
    device: torch.device,
    cfg: TrainConfig,
    save_dir: Path,
    visualizer: Visualizer,
) -> None:
    model.eval()

    with torch.no_grad():
        batch = next(iter(val_loader))
        imgs = batch["image"].to(device)
        targets = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        preds = model(imgs, cmd_onehot=targets["command"])
        detections = non_max_suppression(
            preds["bboxes"], conf_thres=cfg.conf_thres, nc=cfg.nc
        )

        num_samples = min(len(imgs), cfg.num_vis_samples)
        for i in range(num_samples):
            grid = visualizer.create_comparison_grid(i, targets, preds, detections)
            cv2.imwrite(str(save_dir / f"sample_{i:02d}.jpg"), grid)

    model.train()
    logger.info("Saved %d visualization samples to '%s'", num_samples, save_dir)


def train(cfg: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    run_dir = Path(cfg.save_dir)
    vis_dir = run_dir / "visualizations"
    ckpt_dir = run_dir / "checkpoints"
    for d in (vis_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    data_cfg = load_config()
    data_cfg.data.dataset_yaml = cfg.dataset_yaml
    data_cfg.data.batch_size = cfg.batch_size
    data_cfg.data.image_size = cfg.image_size
    data_cfg.data.num_workers = cfg.num_workers

    # Configure Light Augmentation
    data_cfg.data.augment.mosaic = 0.0
    data_cfg.data.augment.mixup = 0.0
    data_cfg.data.augment.rotate_deg = 2.0
    data_cfg.data.augment.scale = 0.1
    data_cfg.data.augment.hsv_v = 0.0
    data_cfg.data.augment.hsv_s = 0.0
    data_cfg.data.augment.color_jitter = 0.0
    data_cfg.data.augment.fliplr = 0.0
    logger.info("Configured Light Augmentation (No Mosaic/Mixup) & Cosine LR")

    # Manual Dataset Creation to allow doubling samples before DataLoader init
    tr_pipe = StandardAugmentor(training=True, imgsz=data_cfg.data.image_size, config=data_cfg.data.augment)
    val_pipe = StandardAugmentor(training=False, imgsz=data_cfg.data.image_size)

    tr_ds = NeuroPilotDataset(
        dataset_yaml=data_cfg.data.dataset_yaml,
        split='train',
        transform=tr_pipe,
        imgsz=data_cfg.data.image_size
    )
    val_ds = NeuroPilotDataset(
        dataset_yaml=data_cfg.data.dataset_yaml,
        split='val',
        transform=val_pipe,
        imgsz=data_cfg.data.image_size
    )

    # Augment dataset size from 1300 -> 2600 by repeating samples BEFORE building Dataloader
    original_size = len(tr_ds.samples)
    tr_ds.samples = tr_ds.samples * 2
    logger.info(
        "Augmented training dataset size: %d -> %d (2600 target)",
        original_size,
        len(tr_ds.samples)
    )

    train_loader = build_dataloader(
        tr_ds,
        batch=data_cfg.data.batch_size,
        shuffle=True,
        workers=data_cfg.data.num_workers,
        collate_fn=custom_collate_fn
    )
    val_loader = build_dataloader(
        val_ds,
        batch=data_cfg.data.batch_size,
        shuffle=False,
        workers=data_cfg.data.num_workers,
        collate_fn=custom_collate_fn
    )

    logger.info(
        "Dataset loaded | train=%d batches | val=%d batches",
        len(train_loader),
        len(val_loader),
    )

    model, optimizer, scheduler = build_model_and_optimizer(cfg, device)
    # Inject real names from dataset
    model.names = getattr(tr_ds, 'names', {i: f"class_{i}" for i in range(cfg.nc)})

    criterion = build_criterion(cfg, model, device)
    visualizer = Visualizer(HeatmapLoss(), image_size=cfg.image_size)

    model.train()
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        train_metrics = EpochMetrics()
        nan_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.epochs}]", ncols=110)
        for batch in pbar:
            imgs = batch["image"].to(device)
            targets = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            preds = model(imgs, cmd_onehot=targets["command"])
            loss_dict = criterion.advanced(preds, targets)
            loss = loss_dict["total"]

            if torch.isnan(loss):
                nan_count += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_metrics.update(loss_dict)
            avgs = train_metrics.averages()
            pbar.set_postfix(
                total=f"{avgs['total']:.3f}",
                box=f"{avgs['box']:.3f}",
                hm=f"{avgs['heatmap']:.3f}",
                traj=f"{avgs['traj']:.3f}",
            )

        scheduler.step()
        avgs = train_metrics.averages()
        log_msg = (
            f"Epoch {epoch + 1} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"train_total={avgs.get('total', 0):.4f} | box={avgs.get('box', 0):.4f} | "
            f"hm={avgs.get('heatmap', 0):.4f} | cls={avgs.get('cls', 0):.4f} | "
            f"traj={avgs.get('traj', 0):.4f}"
        )
        if nan_count > 0:
            log_msg += f" | nan={nan_count}"
        logger.info(log_msg)

        if (epoch + 1) % cfg.val_every_n_epochs == 0:
            val_avgs = run_validation(model, val_loader, criterion, device)
            logger.info(
                "Epoch %d | val_total=%.4f | val_box=%.4f | val_hm=%.4f",
                epoch + 1,
                val_avgs.get("total", 0),
                val_avgs.get("box", 0),
                val_avgs.get("heatmap", 0),
            )

            epoch_vis_dir = vis_dir / f"epoch_{epoch+1:04d}"
            epoch_vis_dir.mkdir(exist_ok=True)
            save_visualizations(model, val_loader, device, cfg, epoch_vis_dir, visualizer)

            save_checkpoint(
                ckpt_dir / "last.pt", epoch + 1, model, optimizer, val_avgs, cfg=data_cfg
            )

            if val_avgs.get("total", float("inf")) < best_val_loss:
                best_val_loss = val_avgs["total"]
                save_checkpoint(
                    ckpt_dir / "best.pt", epoch + 1, model, optimizer, val_avgs, cfg=data_cfg
                )
                logger.info("New best model saved (val_loss=%.4f)", best_val_loss)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    train(TrainConfig())

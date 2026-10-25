"""Tools to look inside trained CNNs: reload a run, extract embeddings (RQ4) and Grad-CAM maps (RQ5)."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from src import config
from src.data import load_dataset
from src.experiments import read_log
from src.models import build_model
from src.splits import load_split
from src.train import channel_stats, get_device, subsample_train, to_input


def load_trained(run_id, device=None):
    """Rebuild the model of a logged run, load its best weights, and recompute its normalisation
    (per-channel mean and SD of the same training cells it was trained on). Returns (model, mean, sd)."""
    device = device or get_device()
    row = read_log().set_index("run_id").loc[run_id]
    settings = json.loads(row["settings"])
    X, y, _ = load_dataset(int(row["n_classes"]))
    train = subsample_train(load_split("train"), y, settings["label_fraction"], int(row["seed"]))
    mean, sd = channel_stats(X[train])
    model = build_model(row["model"], int(row["n_classes"]))
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / f"{run_id}_best.pt", map_location="cpu"))
    return model.to(device).eval(), mean.to(device), sd.to(device)


@torch.no_grad()
def embeddings(model, X_uint8, mean, sd):
    """Penultimate-layer features (model.embed) for uint8 images (N, 2, H, W) -> numpy (N, D)."""
    device = mean.device
    out = []
    for i in range(0, len(X_uint8), config.EVAL_BATCH_SIZE):
        xb = torch.from_numpy(X_uint8[i:i + config.EVAL_BATCH_SIZE]).to(device)
        out.append(model.embed(to_input(xb, mean, sd)).cpu())
    return torch.cat(out).numpy()


def grad_cam(model, x, target_class):
    """Grad-CAM [Selvaraju et al. 2017] for one normalised image x (1, 2, H, W).

    1. Forward pass; a hook stores the activations A (channels k, h, w) of model.last_conv_layer()
       and a second hook stores the gradient of the target class score with respect to A.
    2. Weight of channel k = mean of its gradient over all positions (how much the class score
       goes up when channel k is more active).
    3. Map = ReLU(sum over k of weight_k * A_k): only regions that push the score up are kept.
    4. Upsample the map to the input size (66 x 66) and scale it to 0..1.
    """
    store = {}

    def save_activation(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = model.last_conv_layer().register_forward_hook(save_activation)
    try:
        model.zero_grad()
        scores = model(x)
        scores[0, target_class].backward()
    finally:
        handle.remove()
    act, grad = store["act"][0], store["grad"][0]  # (k, h, w)
    weights = grad.mean(dim=(1, 2))  # (k,)
    cam = F.relu((weights.view(-1, 1, 1) * act).sum(dim=0))
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def border_fraction(cam, width=config.GRADCAM_BORDER_PX):
    """Share of the Grad-CAM heat that falls in the outer `width` pixels of the image (shortcut check)."""
    ring = np.ones(cam.shape, bool)
    ring[width:-width, width:-width] = False
    return float(cam[ring].sum() / cam.sum()) if cam.sum() > 0 else np.nan


def mask_fraction(cam, mask):
    """Share of the Grad-CAM heat that falls inside the brightfield cell mask (src.features.cell_mask).
    Compare with the mask's share of the image area: equal values mean the heat ignores the cell."""
    return float(cam[mask].sum() / cam.sum()) if cam.sum() > 0 else np.nan

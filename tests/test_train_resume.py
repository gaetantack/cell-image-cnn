import torch

from src import config
from src import train


def test_checkpoint_save_and_resume(tmp_path, monkeypatch, capsys):
    """A smoke run writes <run_id>_last.pt; a second run with --resume must load it, including the
    generator state, also on MPS (torch.load(map_location=device) moves that state off the CPU)."""
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args = ["--model", "smallcnn", "--smoke-test", "--device", device]

    train.main(args)
    last = tmp_path / "smoke" / "smallcnn_c4_smoke_s0_last.pt"
    assert last.exists()
    assert torch.load(last, map_location=device, weights_only=False)["epoch"] == 0

    capsys.readouterr()
    m = train.main(args + ["--resume"])
    assert "resumed smallcnn_c4_smoke_s0 at epoch 1" in capsys.readouterr().out
    assert 0.0 <= m["macro_f1"] <= 1.0

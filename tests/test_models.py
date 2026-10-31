import numpy as np
import pytest
import timm
import torch

from src import config
from src.models import ResNet18, SmallCNN, build_model
from src.train import augment_basic, class_weights, subsample_train


@pytest.fixture(scope="module")
def pretrained():
    return ResNet18(n_classes=4, pretrained=True).eval()


@pytest.mark.parametrize("n_classes", [4, 7])
def test_smallcnn_shapes(n_classes):
    model = SmallCNN(n_classes).eval()
    x = torch.randn(3, 2, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert model(x).shape == (3, n_classes)
    assert model.embed(x).shape == (3, config.SMALLCNN_CHANNELS[-1])


@pytest.mark.parametrize("size", [config.IMAGE_SIZE, config.RESNET_INPUT_SIZE])
def test_resnet_shapes_both_input_sizes(size):
    """66 px input is upscaled inside the model; 128 px input is used as is."""
    model = ResNet18(n_classes=7, pretrained=False).eval()
    x = torch.randn(2, 2, size, size)
    assert model(x).shape == (2, 7)
    assert model.embed(x).shape == (2, 512)


def test_resnet_conv1_has_two_input_channels():
    for name in ("resnet18_pretrained", "resnet18_scratch"):
        assert build_model(name, 4).net.conv1.weight.shape == (64, 2, 7, 7)


def test_pretrained_conv1_is_mean_of_rgb(pretrained):
    rgb = timm.create_model("resnet18", pretrained=True).conv1.weight  # (64, 3, 7, 7)
    w = pretrained.net.conv1.weight
    for c in range(2):
        assert torch.allclose(w[:, c], rgb.mean(dim=1))


def test_pretrained_keeps_imagenet_weights_after_conv1(pretrained):
    ref = timm.create_model("resnet18", pretrained=True)
    assert torch.equal(pretrained.net.layer1[0].conv1.weight, ref.layer1[0].conv1.weight)


def test_class_weights_balance_the_classes():
    y = np.array([0] * 6 + [1] * 3 + [2])
    w = class_weights(y, 3).numpy()
    counts = np.bincount(y)
    assert np.allclose(w * counts, w[0] * counts[0])  # every class has the same total weight


def test_subsample_train_is_stratified_and_keeps_every_class():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    train = np.arange(1000)
    keep = subsample_train(train, y, 0.05, seed=0)
    assert set(y[keep]) == {0, 1, 2}
    assert 40 <= len(keep) <= 55
    assert np.array_equal(keep, subsample_train(train, y, 0.05, seed=0))  # seeded


def test_augment_keeps_shape_and_pixel_content():
    x = torch.randn(8, 2, 66, 66)
    gen = torch.Generator().manual_seed(0)
    out = augment_basic(x, gen)
    assert out.shape == x.shape
    # rotations and flips only move pixels; jitter changes values by at most ~10%
    assert torch.allclose(out.abs().mean(), x.abs().mean(), rtol=0.2)


@pytest.mark.skipif(not config.CACHE_FILE.exists(), reason="image cache not present")
@pytest.mark.parametrize("model", ["smallcnn", "resnet18_scratch"])
def test_train_smoke_run_finishes(model):
    from src.train import main
    m = main(["--model", model, "--smoke-test", "--device", "cpu"])
    assert 0 <= m["macro_f1"] <= 1


def test_epoch_order_repeats_small_training_sets():
    from src.train import epoch_order
    y = np.array([0, 0, 0, 1, 1, 2])
    gen = torch.Generator().manual_seed(0)
    order = epoch_order(y, "weighted", gen, n_draws=15)
    assert len(order) == 15
    assert sorted(order[:6].tolist()) == list(range(6))  # first pass = each cell once
    assert len(epoch_order(y, "oversample", gen, n_draws=15)) == 15

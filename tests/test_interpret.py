import numpy as np
import pytest
import torch

from src.interpret import border_fraction, grad_cam, mask_fraction
from src.models import ResNet18, SmallCNN


@pytest.mark.parametrize("model", [SmallCNN(4), ResNet18(4, pretrained=False)])
def test_grad_cam_shape_and_range(model):
    torch.manual_seed(0)
    model.eval()
    cam = grad_cam(model, torch.randn(1, 2, 66, 66), target_class=1)
    assert cam.shape == (66, 66)
    assert cam.min() >= 0 and cam.max() <= 1


def test_heat_fractions():
    cam = np.zeros((66, 66))
    cam[30:36, 30:36] = 1  # heat only in the centre
    mask = np.zeros((66, 66), bool)
    mask[20:46, 20:46] = True
    assert mask_fraction(cam, mask) == 1.0
    assert border_fraction(cam) == 0.0
    cam[0, 0] = 36  # as much heat in one corner pixel as in the centre
    assert border_fraction(cam) == pytest.approx(0.5)

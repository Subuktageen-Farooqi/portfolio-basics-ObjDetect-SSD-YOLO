import torch

from object_detection_ssd_yolo_pytorch import SSDModel, YOLOModel


def test_pytorch_dummy_shapes():
    with torch.no_grad():
        ssd = SSDModel(num_classes=21, num_boxes=6).eval()
        yolo = YOLOModel(num_classes=20, num_boxes=3).eval()

        ssd_class, ssd_box = ssd(torch.randn(1, 3, 300, 300))
        yolo_pred = yolo(torch.randn(1, 3, 416, 416))

        assert ssd_class.shape[0] == 1
        assert ssd_class.shape[-1] == 21
        assert ssd_box.shape[-1] == 4
        assert yolo_pred.shape[1] == 3 * (5 + 20)

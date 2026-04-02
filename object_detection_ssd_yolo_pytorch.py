import torch
import torch.nn as nn


# =========================
# SSD
# =========================


class SSDBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


class SSDHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_boxes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.class_conv = nn.Conv2d(in_channels, num_boxes * num_classes, kernel_size=3, padding=1)
        self.box_conv = nn.Conv2d(in_channels, num_boxes * 4, kernel_size=3, padding=1)

    def forward(self, x):
        b = x.shape[0]
        class_logits = self.class_conv(x)
        box_reg = self.box_conv(x)

        class_logits = class_logits.permute(0, 2, 3, 1).contiguous()
        class_logits = class_logits.view(b, -1, self.num_classes)

        box_reg = box_reg.permute(0, 2, 3, 1).contiguous()
        box_reg = box_reg.view(b, -1, 4)
        return class_logits, box_reg


class SSDModel(nn.Module):
    def __init__(self, num_classes: int = 21, num_boxes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.backbone = SSDBackbone()

        # Reconstructed: matching the tutorial's 3-scale detection idea in PyTorch.
        self.heads = nn.ModuleList([SSDHead(128, num_classes, num_boxes) for _ in range(3)])
        self.downsample = nn.ModuleList(
            [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) for _ in range(2)]
        )

    def forward(self, x):
        x = self.backbone(x)
        class_outputs, box_outputs = [], []

        cls_0, box_0 = self.heads[0](x)
        class_outputs.append(cls_0)
        box_outputs.append(box_0)

        x = self.downsample[0](x)
        cls_1, box_1 = self.heads[1](x)
        class_outputs.append(cls_1)
        box_outputs.append(box_1)

        x = self.downsample[1](x)
        cls_2, box_2 = self.heads[2](x)
        class_outputs.append(cls_2)
        box_outputs.append(box_2)

        classes_concat = torch.cat(class_outputs, dim=1)
        boxes_concat = torch.cat(box_outputs, dim=1)
        return classes_concat, boxes_concat


# =========================
# YOLO
# =========================


class YOLOBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.features(x)


class YOLOHead(nn.Module):
    def __init__(self, in_channels: int = 512, num_classes: int = 20, num_boxes: int = 3):
        super().__init__()
        self.output_channels = num_boxes * (5 + num_classes)
        # Reconstructed: single 1x1 conv prediction layer equivalent to the TF tutorial head.
        self.pred = nn.Conv2d(in_channels, self.output_channels, kernel_size=1)

    def forward(self, x):
        return self.pred(x)


class YOLOModel(nn.Module):
    def __init__(self, num_classes: int = 20, num_boxes: int = 3):
        super().__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(in_channels=512, num_classes=num_classes, num_boxes=num_boxes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssd_num_classes = 21
    ssd_num_boxes = 6
    yolo_num_classes = 20
    yolo_num_boxes = 3

    ssd_model = SSDModel(num_classes=ssd_num_classes, num_boxes=ssd_num_boxes).to(device)
    yolo_model = YOLOModel(num_classes=yolo_num_classes, num_boxes=yolo_num_boxes).to(device)

    with torch.no_grad():
        ssd_input = torch.randn(1, 3, 300, 300, device=device)
        ssd_class_pred, ssd_box_pred = ssd_model(ssd_input)
        print("SSD Class Prediction Shape:", tuple(ssd_class_pred.shape))
        print("SSD Box Prediction Shape:", tuple(ssd_box_pred.shape))

        yolo_input = torch.randn(1, 3, 416, 416, device=device)
        yolo_pred = yolo_model(yolo_input)
        print("YOLO Prediction Shape:", tuple(yolo_pred.shape))

        assert ssd_class_pred.shape[-1] == ssd_num_classes
        assert ssd_box_pred.shape[-1] == 4
        assert yolo_pred.shape[1] == yolo_num_boxes * (5 + yolo_num_classes)

    print("All PyTorch shape checks passed.")

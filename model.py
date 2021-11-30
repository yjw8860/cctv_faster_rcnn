import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import math


def faster_rcnn_resnet_50():
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    return model

def retinanet_resnet_50(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    # in_features = model.head.classification_head.conv[0].in_channels
    # num_anchors = model.head.classification_head.num_anchors
    # model.head.classification_head.num_classes = num_classes
    #
    # cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    # torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    # torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
    # model.head.classification_head.cls_logits = cls_logits

    return model

def ssd_vgg16(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

    num_columns = model.head.classification_head.num_columns
    cls_logits = torch.nn.Conv2d(256, num_columns * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
    model.head.classification_head.cls_logits = cls_logits

    return model
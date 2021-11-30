import torch
import math
import sys
import os
import pandas as pd
from tqdm import tqdm
import re
import pyscreenshot as ImageGrab
import json

from cctv_dataset import cctvDataset
from ops import collate_fn, MetricLogger, reduce_dict, SmoothedValue, AP, get_mAP
from model import faster_rcnn_resnet_50



def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    model_name = 'faster_rcnn'
    num_classes = 2
    data_root = 'D:/DATA/cctv_211021'
    dataset_list = os.listdir(data_root)

    batch_size = 16

    # let's train it for 10 epochs
    num_epochs = 10
    save_dir = './results'
    img_save_dir = './screenshot'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)
    for d, dataset_name in enumerate(dataset_list):
        save_name = re.sub('[/]', '_', dataset_name)
        save_path = f'./results/{save_name}_{model_name}.pth'
        img_save_path = f'./screenshot/{model_name}_{save_name}.png'
        # # get the model using our helper function
        model = faster_rcnn_resnet_50(num_classes)
        # move model to the right device
        model.to(device)
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        # use our dataset and defined transformations
        dataset = cctvDataset(data_root, dataset_name, is_train=True, is_valid=False)
        dataset_valid = cctvDataset(data_root, dataset_name, is_train=False, is_valid=True)
        dataset_test = cctvDataset(data_root, dataset_name, is_train=False, is_valid=False)
        optimizer = torch.optim.Adam(params)

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            collate_fn=collate_fn)

        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batch_size, shuffle=True, num_workers=0,
            collate_fn=collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=collate_fn)
        pre_mAP = 0
        for epoch in range(num_epochs):
            model.train()
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
            header = "Epoch: [{}]".format(epoch)
            epoch_loss = 0
            for imgs, targets in metric_logger.log_every(data_loader, print_freq=1, header=header):
                imgs = list(img.to(device) for img in imgs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()
                epoch_loss += losses
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # checkpoint = torch.load(save_path)
            # model.load_state_dict(checkpoint)
            ground_truths = []
            detections = []
            classes = []

            print('---------------Calculate mAP---------------')
            with torch.no_grad():
                for imgs, targets in tqdm(data_loader_valid):
                    model.eval()
                    imgs = list(img.to(device) for img in imgs)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    predictions = model(imgs)
                    for prediction, target in zip(predictions, targets):
                        boxes, labels, img_id = target['boxes'], target['labels'], target['image_id']
                        boxes = boxes.cpu().detach().numpy()
                        labels = labels.cpu().detach().numpy()
                        img_id = img_id.cpu().detach().numpy()
                        for box, label in zip(boxes, labels):
                            ground_truths.append([str(img_id[0]), label, 1, tuple(box)])
                            classes.append(label)
                        boxes_, labels_, scores_ = prediction['boxes'], prediction['labels'], prediction['scores']
                        boxes_ = boxes_.cpu().detach().numpy()
                        labels_ = labels_.cpu().detach().numpy()
                        scores_ = scores_.cpu().detach().numpy()
                        for box_, label_, score_ in zip(boxes_, labels_, scores_):
                            detections.append([str(img_id[0]), label_, score_, tuple(box_)])
            classes = list(set(classes))
            classes.sort()
            result = AP(detections, ground_truths, classes)
            mAP = get_mAP(result)
            if epoch == 0:
                pre_mAP = mAP
                torch.save(model.state_dict(), save_path)
            else:
                if pre_mAP < mAP:
                    torch.save(model.state_dict(), save_path)
                    pre_mAP = mAP
            print(f'Epoch:{epoch} | validation mAP:{mAP}')
        check_point = torch.load(save_path)
        model.load_state_dict(check_point)
        ground_truths = []
        detections = []
        classes = []
        with torch.no_grad():
            for imgs, targets in tqdm(data_loader_test):
                model.eval()
                imgs = list(img.to(device) for img in imgs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(imgs)
                for prediction, target in zip(predictions, targets):
                    boxes, labels, img_id = target['boxes'], target['labels'], target['image_id']
                    boxes = boxes.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    img_id = img_id.cpu().detach().numpy()
                    for box, label in zip(boxes, labels):
                        ground_truths.append([str(img_id[0]), label, 1, tuple(box)])
                        classes.append(label)
                    boxes_, labels_, scores_ = prediction['boxes'], prediction['labels'], prediction['scores']
                    boxes_ = boxes_.cpu().detach().numpy()
                    labels_ = labels_.cpu().detach().numpy()
                    scores_ = scores_.cpu().detach().numpy()
                    for box_, label_, score_ in zip(boxes_, labels_, scores_):
                        detections.append([str(img_id[0]), label_, score_, tuple(box_)])
        classes = list(set(classes))
        classes.sort()
        result = AP(detections, ground_truths, classes)
        print(result)
        mAP = get_mAP(result)
        print(f'test mAP:{mAP}')
        df = pd.DataFrame({"dataset_name": dataset_name, 'mAP': mAP}, columns=['dataset_name', 'mAP'], index=[0])
        if d == 0:
            df.to_csv(f'./{model_name}_results.csv', index=False, mode='w')
        else:
            df.to_csv(f'./{model_name}_results.csv', index=False, mode='a', header=False)
        img = ImageGrab.grab()
        img.save(img_save_path)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
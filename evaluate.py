#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:48:45 2022

@author: raju
"""
import argparse
import os
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import torchvision.transforms as transforms
import yaml
from munch import Munch
from models import MV_DEFEAT_ddsm
from dataset import DDSM_dataset
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import AverageMeter, accuracy
from torchmetrics.classification import Accuracy

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("On which device we are on:{}".format(device))


def evaluate_multi(cfg, test_loader, model, model_checkpoint_path, device):
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    targets = [[] for _ in range(cfg.arch.num_classes)]
    probs = [[] for _ in range(cfg.arch.num_classes)]
    pred_labels = [[] for _ in range(cfg.arch.num_classes)]

    accuracies = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    f1_value = AverageMeter()

    with torch.no_grad():
        model.eval()
        for input1, input2, input3, input4, target in tqdm(test_loader):
            input1, input2, input3, input4, target = input1.to(device), input2.to(device), input3.to(device), input4.to(
                device), target.to(device)
            prob = model(input1, input2, input3, input4)
            label = torch.argmax(prob, 1)

            acc = accuracy(prob, target)
            prec = precision_score(label.cpu().numpy(), target.cpu().numpy(), average='weighted')
            rec = recall_score(label.cpu().numpy(), target.cpu().numpy(), average='weighted')
            f1 = f1_score(label.cpu().numpy(), target.cpu().numpy(), average='weighted')

            accuracies.update(acc, input1.size(0))
            precision.update(prec, input1.size(0))
            recall.update(rec, input1.size(0))
            f1_value.update(f1, input1.size(0))

            for i in range(cfg.arch.num_classes):
                targets[i].extend(target.cpu().numpy() == i)
                probs[i].extend(prob[:, i].cpu().numpy())
                pred_labels[i].extend(label.cpu().numpy() == i)

    return targets, probs, pred_labels, accuracies, precision, recall, f1_value


def normalize_array(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def main(cfg):
    checkpoint_dir = os.path.join(cfg.training.checkpoints_dir, '{}_{}'.format(cfg.data.dataset_name, cfg.data.task),
                                  cfg.training.fusion_type, cfg.data.analysis)
    checkpoint_path = os.path.join(checkpoint_dir, 'epoch_20.pth.tar')

    model = MV_DEFEAT_ddsm(cfg).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train__valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if cfg.data.dataset_name == 'DDSM':
        test_dataset = DDSM_dataset(cfg.data.root, view_laterality=cfg.data.laterality,
                                          split='test', transform=train__valid_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                              num_workers=cfg.data.workers, pin_memory=True)

    targets, probs, pred_labels, accuracies, precision, recall, f1, = evaluate_multi(cfg, test_loader, model,
                                                                                     checkpoint_path, device)

    import numpy as np
    # https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html
    def confidence_interval(value, n_samples):
        return 1.96 * np.sqrt((value * (1 - value)) / n_samples)

    print(r'Precision {0:.3f} $\pm$ {1:.3f}'.format(precision.avg * 100,confidence_interval(precision.avg, len(test_dataset))))
    print(r'Recall {0:.3f} $\pm$ {1:.3f}'.format(recall.avg * 100, confidence_interval(recall.avg, len(test_dataset))))
    print(r'F1-score {0:.3f} $\pm$ {1:.3f}'.format(f1.avg * 100, confidence_interval(f1.avg, len(test_dataset))))
    print(r'Accuracy {0:.3f} $\pm$ {1:.3f}'.format(accuracies.avg.cpu().numpy(), confidence_interval(accuracies.avg.cpu().numpy() / 100,
                                                                       len(test_dataset))))

    #plot_multi_roc_curve(targets, probs, args.class_names, cfg.data.dataset_name)

    for i in range(cfg.arch.num_classes):
        print('class {}'.format(args.class_names[i]))
        auc = roc_auc_score(targets[i], probs[i])
        print('AUC: {0:.4f} $\pm$ {1:.3f}'.format(auc, confidence_interval(auc, len(test_dataset))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='ddsm_ipsilateral_config.yml')
    # parser.add_argument('--class_names', type=list, default=['BIRADS 1', 'BIRADS 2', 'BIRADS 3', 'BIRADS 4', 'BIRADS 5'])
    parser.add_argument('--class_names', type=list, default=['Density A', 'Density B', 'Density C', 'Density D'])
    # parser.add_argument('--class_names', type=list, default=['Benign', 'Malignant', 'Normal'])
    # parser.add_argument('--class_names', type=list, default=['Benign', 'Malignant'])

    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        cfg = Munch.fromYAML(f)
    main(cfg)
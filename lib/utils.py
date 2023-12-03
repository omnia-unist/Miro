
import datetime
import logging
import os
import warnings

import torch
from torch.nn import functional as F
from torchvision import transforms
#str
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
        
        
#str function
#XINYUE 
def get_hardness(train_dataset, classes_so_far, model, filename, device):
    dataloader = DataLoader(train_dataset, batch_size=128, num_workers=3, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    losses = {x : [] for x in range(classes_so_far)}
    entropies = {x : [] for x in range(classes_so_far)}
    scores = {x : [] for x in range(classes_so_far)}
    flips = {x : [] for x in range(classes_so_far)}
    filenames = {x : [] for x in range(classes_so_far)}
    pred_labels = {x : [] for x in range(classes_so_far)}
    prob_labels = {x : [] for x in range(classes_so_far)}



    model.eval()
    for i, (inputs, targets, filename) in enumerate(dataloader):
        if i % 100 == 0:
            print(f"calculating hardness...{i}")
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)

        # get loss
        loss_value = criterion(outputs, targets)

        # get entropy
        pred = torch.nn.functional.softmax(outputs, dim=1).detach().cpu()
        flip = (torch.max(pred, dim=1)[1] == targets.cpu()).squeeze().long()
        #1 if targets.cpu() == torch.max(pred, dim=1)[1] else 0

        entropy = (np.log(pred) * pred).sum(axis=1) * -1.
        #entropy = gradtail(outputs).detach().cpu()
        score = 0.5 * (flip * (entropy/classes_so_far) + (1-flip) * (2-(entropy/classes_so_far)) )

        for t in range(classes_so_far):
            #print(targets)
            index_t = (targets.cpu() == t).squeeze().nonzero(as_tuple=True)[0]

            #filenames[t].extend([filename[i] for i in index_t]) assumes 
            filenames[t].extend([filename[i] for i in index_t])
            flips[t].extend(flip[index_t].tolist())
            losses[t].extend(loss_value[index_t].tolist())
            entropies[t].extend(entropy[index_t].tolist())
            scores[t].extend(score[index_t].tolist())
            # if(len(index_t)>0 and len(index_t)<127):
            #     print(filenames[t])
            #     print(flips[t])
    model.train()
    # print("First 10 targets: " + str(targets[0:10]))
    # print("First 10 flips: " + str(flips[0][0:10]))
    # print("First 10 predictions: " + str(pred_labels[0:10]))
    # print("First 10 probs: " + str(prob_labels[0:10]))
    # print("Pretrained accuracy: " + str(sum(flips)))


    # sys.stdout.flush()


    # import csv

    # csvfile = open(filename, "w", newline="")
    # csv_writer = csv.writer(csvfile)
    
    # csv_writer.writerow(["sample #","target","predicted label","probability","flip","loss","entropy","score"])
    # for i, (f,l,e,s) in enumerate(zip(flips,losses, entropies, scores)):
    #     csv_writer.writerow([i,f,l,e,s])
    # csvfile.close()
    return losses, entropies, scores, flips, filenames


def extract_features(model, loader, device):

    targets, features = [], []

    state = model.training
    model.eval()


    for inputs, _targets in loader:
        #print(inputs, _targets)
        _targets = _targets.numpy()
        #_features = model.features(inputs.to(device)).detach().cpu().numpy()
        _features = model.feature_extractor(inputs.to(device)).detach().cpu().numpy()
        
        features.append(_features)
        targets.append(_targets)

    model.train(state)

    #print(targets)
    
    return np.concatenate(features), np.concatenate(targets)


def Image_transform(images, transform):
    data = transform(images[0]).unsqueeze(0)
    for index in range(1, len(images)):
        data = torch.cat((data, transform(images[index]).unsqueeze(0)), dim=0)
    return data


def compute_class_mean(images, model, device, transform):
    x = Image_transform(images, transform).to(device)

    with torch.no_grad():
        feature_extractor_output = F.normalize(model.feature_extractor(x).detach()).cpu().numpy()
        #feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
    class_mean = np.mean(feature_extractor_output, axis=0)
    return class_mean, feature_extractor_output


def compute_exemplar_class_mean(classes_so_far, data, model, device, base_transform, classify_transform=None):
    class_mean_set = []
    for index in range(classes_so_far):
        print("compute the class mean of %s"%(str(index)))
        exemplar, exp_idx = data.get_sub_data(index)
        #exemplar = data_dict[index]
        print("the number of examplar : ", len(exemplar))
        
        class_mean, _ = compute_class_mean(exemplar, model, device, base_transform)
        class_mean_,_= compute_class_mean(exemplar, model, device, classify_transform)
        class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
        #class_mean = class_mean/np.linalg.norm(class_mean)
        
        class_mean_set.append(class_mean)
    return class_mean_set


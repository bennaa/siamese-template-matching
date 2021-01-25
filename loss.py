import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss = torch.mean(0.5 *
                          ((1 - label) * torch.pow(distance, 2) +
                           label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
                          )

        return loss, distance


class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.pow(F.pairwise_distance(anchor, positive, keepdim=True), 2)
        negative_distance = torch.pow(F.pairwise_distance(anchor, negative, keepdim=True), 2)

        loss = torch.mean(F.relu(self.margin + positive_distance - negative_distance))
        return loss, (positive_distance, negative_distance)

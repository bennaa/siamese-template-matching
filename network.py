import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SiameseNetwork(nn.Module):
    def __init__(self, channels=3):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(3200, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 256),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

    def predict(self, input1, input2, threshold):
        output1, output2 = self(input1.to(device), input2.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2).detach().cpu().numpy()

        # 0: same class (<= threshold), 1: different class (> threshold)
        # reshape because from (n,) to (n,1) (needed to compare labels later)
        prediction = np.where(euclidean_distance > threshold, 1, 0)
        prediction = prediction.reshape((len(euclidean_distance), 1))

        return prediction, euclidean_distance

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def checkpoint(self, checkpoint_path, threshold, margin, epoch):
        path = f"{checkpoint_path}/" \
               f"_data{datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f')}" \
               f"_threshold{threshold}" \
               f"_margin{margin}" \
               f"_epoch{epoch}.tar"

        torch.save({'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'threshold': threshold,
                    'margin': margin}, path)

        print("\nCheckpoint saved: {}".format(path))


class TripletSiameseNetwork(nn.Module):
    def __init__(self, channels=3):
        super(TripletSiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(3200, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 256),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

    def forward(self, anchor, positive, negative):
        out1 = self.forward_one(anchor)
        out2 = self.forward_one(positive)
        out3 = self.forward_one(negative)
        return out1, out2, out3

    def predict(self, input1, input2, threshold):
        output1, output2, _ = self(input1.to(device), input2.to(device), input2.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2).detach().cpu().numpy()

        # 0: same class (<= threshold), 1: different class (> threshold)
        # reshape because from (n,) to (n,1) (needed to compare labels later)
        prediction = np.where(euclidean_distance > threshold, 1, 0)
        prediction = prediction.reshape((len(euclidean_distance), 1))

        return prediction, euclidean_distance

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def checkpoint(self, checkpoint_path, threshold, margin, epoch):
        path = f"{checkpoint_path}/" \
               f"_data{datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f')}" \
               f"_threshold{threshold}" \
               f"_margin{margin}" \
               f"_epoch{epoch}.tar"

        torch.save({'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'threshold': threshold,
                    'margin': margin}, path)

        print("\nCheckpoint saved: {}".format(path))

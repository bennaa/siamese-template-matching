import datetime
import os
import time

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import utils
from dataset import get_loader
from loss import TripletLoss
from network import TripletSiameseNetwork

torch.manual_seed(1)
np.random.seed(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def evaluate(net, loader, threshold, margin, display=True, display_step=10, epoch=0, type='Validation'):
    with torch.no_grad():
        correct_predictions = 0
        tot = 0

        for i, data in enumerate(loader, 0):
            (anchor, idx1), (positive, idx2), (negative, idx3) = data

            positive_prediction, positive_distance = net.predict(anchor, positive, threshold)
            correct_predictions += np.sum(np.where(0.0 == positive_prediction, 1, 0))

            negative_prediction, negative_distance = net.predict(anchor, negative, threshold)
            correct_predictions += np.sum(np.where(1.0 == negative_prediction, 1, 0))

            tot += 2 * len(anchor)

            if display and i % display_step == 0:
                utils.plot_pairs(img1=anchor, img2=positive, idx1=idx1, idx2=idx2,
                                 labels=torch.from_numpy(np.zeros(len(anchor))), distances=positive_distance,
                                 predictions=positive_prediction)
                utils.plot_pairs(img1=anchor, img2=negative, idx1=idx1, idx2=idx3,
                                 labels=torch.from_numpy(np.ones(len(anchor))), distances=negative_distance,
                                 predictions=negative_prediction)

                # for i in range(len(anchor)):
                #     print("Positive ({}, {}): Distance = {:<.4f} | Prediction = {} (expected 0)".format(idx1[i], idx2[i], positive_distance[i], positive_prediction[i]))
                #     print("Negative ({}, {}): Distance = {:<.4f} | Prediction = {} (expected 1)\n".format(idx1[i], idx3[i], negative_distance[i], negative_prediction[i]))

        accuracy = correct_predictions / tot
        if epoch is not None:
            comment = "{} at epoch {:3d} = " \
                      "threshold = {:<.2f} | " \
                      "margin = {:<.2f} | " \
                      "accuracy = {:<.3f}\n".format(type, epoch + 1, threshold, margin, accuracy)
        else:
            comment = "{}: " \
                      "threshold = {:<.2f} | " \
                      "margin = {:<.2f} | " \
                      "accuracy = {:<.3f}\n\n".format(type, threshold, margin, accuracy)
        print(comment)
        utils.save_comments(checkpoint_path,
                            comment=f"Triplet => Date={datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f')} | {comment}")
    return accuracy


def train(net,
          loss_fn,
          trainloader, validationloader,
          threshold,
          margin,
          experiment_description,
          lr=0.001,
          save_step=0, save_path=None,
          start_epoch=0,
          log_step=200):
    globaliter_train = 0
    writer_train = SummaryWriter(log_dir=f"./log/{experiment_description}/train")
    writer_test = SummaryWriter(log_dir=f"./log/{experiment_description}/test")

    criterion = loss_fn(margin=margin)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print('Start training....')
    total_time = time.time()

    for epoch in range(start_epoch, epochs):

        start = time.time()

        for i, data in enumerate(trainloader, 0):
            (anchor, idx1), (positive, idx2), (negative, idx3) = data
            optimizer.zero_grad()

            anchor_output, positive_output, negative_output = net(anchor.to(device), positive.to(device),
                                                                  negative.to(device))
            loss, (positive_distance, negative_distance) = criterion(anchor_output, positive_output, negative_output)

            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                avg_positive_distance, avg_negative_distance = positive_distance.mean(), negative_distance.mean()

                writer_train.add_scalar('loss', loss.item(), global_step=globaliter_train)
                writer_train.add_scalars('distances', {
                    'avg_positive': avg_positive_distance,
                    'avg_negative': avg_negative_distance
                }, global_step=globaliter_train)

                globaliter_train += 1

                print(
                    "Epoch={:<3d} | Batch={:<5d} | Avg positive dist={:<.3f} | Avg negative dist={:<.3f} | Loss={:<5f}".format(
                        epoch + 1, i, avg_positive_distance, avg_negative_distance, loss.item()))

        accuracy = evaluate(net=net, loader=validationloader,
                            threshold=threshold, margin=margin,
                            display=False, epoch=epoch)

        writer_test.add_scalar('accuracy', accuracy, global_step=epoch)

        # save checkpoint
        if save_path and save_step > 0 and epoch > 0 and (epoch + 1) % save_step == 0:
            net.checkpoint(checkpoint_path=save_path, threshold=threshold, margin=margin, epoch=epoch + 1)

        print(f"Epoch finished in {datetime.timedelta(seconds=int(time.time() - start))}\n")

    print(f"Training Finished in {datetime.timedelta(seconds=int(time.time() - total_time))}!!\n")
    net.checkpoint(checkpoint_path=save_path, threshold=threshold, margin=margin, epoch=epochs)

    writer_train.close()
    writer_test.close()


def experiments(trainloader, validationloader, testloader):
    # thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    thresholds = [2.5, 3]
    margins = [2, 2.5]

    for threshold in thresholds:
        for margin in margins:
            experiment_description = f"Triple_threshold{threshold}_margin{margin}"
            print(f"Start experiment with threshold = {threshold} | margin = {margin}")
            net = TripletSiameseNetwork().to(device)
            train(net=net, loss_fn=TripletLoss,
                  threshold=threshold, margin=margin,
                  save_path=checkpoint_path, experiment_description=experiment_description,
                  trainloader=trainloader, validationloader=validationloader)
            evaluate(net, testloader, threshold, margin, display=False, type='Testing')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    checkpoint_path = 'models'
    PREFIX = 'fruits'

    # LOADERS
    trainloader = get_loader(f'dataset/training_dataset_{PREFIX}.pt', shuffle=False, batch_size=32, triplet=True)
    validationloader = get_loader(f'dataset/validation_dataset_{PREFIX}.pt', shuffle=False, batch_size=8, triplet=True)
    testloader = get_loader(f'dataset/testing_dataset_{PREFIX}.pt', shuffle=False, batch_size=8, triplet=True)

    print("\nDevice used: {}\n".format(device))

    experiments(trainloader, validationloader, testloader)

import datetime
import os
import time

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import utils
from dataset import get_loader
from loss import ContrastiveLoss
from network import SiameseNetwork

torch.manual_seed(1)
np.random.seed(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def evaluate(net, loader, threshold, margin, display=True, display_step=1, epoch=None, type='Validation'):
    with torch.no_grad():
        correct_predictions = 0
        tot = 0

        for i, data in enumerate(loader, 0):
            (input1, idx1), (input2, idx2), labels = data

            prediction, euclidean_distance = net.predict(input1, input2, threshold)
            correct_predictions += np.sum(np.where(labels.numpy() == prediction, 1, 0))
            tot += len(input1)

            if display and i % display_step == 0:
                utils.plot_pairs(img1=input1, img2=input2, idx1=idx1, idx2=idx2, labels=labels,
                                 distances=euclidean_distance, predictions=prediction)

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

        utils.save_comments(checkpoint_path,
                            comment=f"Date={datetime.datetime.now().strftime('%Y%m%d_%H%M_%S_%f')} | {comment}")
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
            (input1, idx1), (input2, idx2), labels = data
            optimizer.zero_grad()

            output1, output2 = net(input1.to(device), input2.to(device))
            loss, distance = criterion(output1, output2, labels.to(device))

            loss.backward()
            optimizer.step()

            if i % log_step == 0:
                avg_positive_distance, avg_negative_distance = utils.compute_avg_distance_per_class(
                    distance, labels)

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
    thresholds = [0.6]
    margins = [1]

    for threshold in thresholds:
        for margin in margins:
            experiment_description = f"threshold{threshold}_margin{margin}"
            print(f"Start experiment with threshold = {threshold} | margin = {margin}")
            net = SiameseNetwork().to(device)
            train(net=net, loss_fn=ContrastiveLoss,
                  threshold=threshold, margin=margin,
                  save_path=checkpoint_path, experiment_description=experiment_description,
                  trainloader=trainloader, validationloader=validationloader)
            evaluate(net, testloader, threshold, margin, display=True, type='Testing')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5
    checkpoint_path = 'models'
    PREFIX = 'fruits'

    # LOADERS
    trainloader = get_loader(f'dataset/training_dataset_{PREFIX}.pt', shuffle=False, batch_size=32)
    validationloader = get_loader(f'dataset/validation_dataset_{PREFIX}.pt', shuffle=False, batch_size=8)
    testloader = get_loader(f'dataset/testing_dataset_{PREFIX}.pt', shuffle=False, batch_size=8)

    print("\nDevice used: {}\n".format(device))

    experiments(trainloader, validationloader, testloader)

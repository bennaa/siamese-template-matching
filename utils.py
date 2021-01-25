import datetime
import os
import pathlib
import time
from collections import defaultdict

import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont

np.random.seed(1)
# ia.seed(4)


# ======================================== HANDLE PATCHES AND BOUNDING BOXES ========================================
def get_patches(image, rows=3, cols=3, target_shape=(100, 100)):
    height, width = image.shape[0] // rows, image.shape[1] // cols
    print(f'PATCH: height={height}, width={width}')

    patches = []
    for row in range(rows):
        for col in range(cols):
            patch = image[row * height:(row + 1) * height, col * width:(col + 1) * width, :]
            patch = Image.fromarray(patch)
            resize_patch = ImageOps.fit(patch, target_shape, Image.ANTIALIAS)
            resize_patch.format = patch.format

            bb = [
                (col * width, row * height),  # x1,y1
                ((col + 1) * width, (row + 1) * height)  # x2,y2
            ]
            patches.append((resize_patch, bb))
    return patches


def draw_bb(image, bb, distance, color):
    (x1, y1), (x2, y2) = bb
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=10)
    draw.text([x1 + 10, y1 + 10], text=distance)#, font=ImageFont.truetype("arial.ttf", 50), fill='green')


def get_pairs(template, patches):
    pairs = [(template, patch) for patch in patches]
    return pairs


def draw_result(image, pairs, predictions, distances, color):
    for idx, (img1, (img2, bb)) in enumerate(pairs):
        prediction = predictions[idx]
        distance = str(round(distances[idx].item(), 3))

        if prediction == 0.0:
            draw_bb(image, bb, distance, color)
    # image.show()


# ======================================== HANDLE DATASET AND PLOTTING ========================================
# TODO: capire se fare data augmentation (per ora non la facciamo)
# def augment(image):
#     augmenters = [
#         iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
#         # iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
#         # iaa.MultiplyBrightness((0.5, 1.5)),
#         iaa.SaltAndPepper(0.1)
#     ]
#     augmenter = random.choice(augmenters)
#     return augmenter(images=image)


# def preprocess(root_dir):
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith('.jpg'):
#                 src = os.path.join(root, file)
#                 dst = os.path.join(root, f"a_{file}")
#                 image = imageio.imread(src)
#                 augmented_image = augment(image)
#
#                 imageio.imwrite(dst, augmented_image)
#                 print(f"File saved: {dst}/{file}")

# NOT USED (but helpful)
def resize_images(root_dir, out_dir='all_resized', target_shape=(105, 105)):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                src = os.path.join(root, file)  # source filename
                dst = root.replace('all', out_dir)  # replace folder name from 'png' to 'sketches'
                dst = dst.replace(' ', '_')

                pathlib.Path(dst).mkdir(parents=True, exist_ok=True)  # create folders (if not exists)
                image = Image.open(src)
                resize_image = ImageOps.fit(image, target_shape, Image.ANTIALIAS)
                resize_image.format = image.format
                resize_image.save(f"{dst}/{file}")
                print(f"File saved: {dst}/{file}")


# USED
def save_comments(path, comment):
    with open('{}/comments.txt'.format(path), 'a') as file:
        file.write(comment)


# USED
def compute_avg_distance_per_class(euclidean_distance, labels):
    euclidean_distance = euclidean_distance.detach().cpu().numpy()
    labels = labels.numpy()

    positive_labels_idx = np.where(labels[:, 0] == 0)
    negative_labels_idx = np.where(labels[:, 0] != 0)

    avg_positive_distance = np.mean(euclidean_distance[positive_labels_idx])
    avg_negative_distance = np.mean(euclidean_distance[negative_labels_idx])

    return avg_positive_distance, avg_negative_distance


# USED
def plot_pairs(img1, img2, labels, idx1, idx2, distances=[], predictions=[], color_scale='RGB'):
    if len(distances) == 0:
        distances = [None] * len(img1)

    if len(predictions) == 0:
        predictions = [None] * len(img1)

    # convert images in the correct shape (original shape=(batch_size, channels, H, W)
    # convert to int with values in (0, 255)
    if color_scale == 'RGB':
        img1 = (img1.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        img2 = (img2.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    else:
        img1 = (torch.squeeze(img1).numpy() * 255).astype(np.uint8)
        img2 = (torch.squeeze(img2).numpy() * 255).astype(np.uint8)

    # stack horizontally each pair, and then stack them vertically
    imgs = np.vstack(([np.hstack((i1, i2)) for (i1, i2) in zip(img1, img2)]))

    # compute the distance between the text (centered relative to the pair)
    offset = imgs.shape[0] // len(labels)

    fontsize = 8
    imgs = Image.fromarray(imgs)
    width, _ = imgs.size

    for i in range(len(labels)):

        id1 = idx1[i].item()
        id2 = idx2[i].item()
        distance = distances[i]
        predicted = predictions[i]
        label = labels[i].item()

        text = f"Classes=({id1}, {id2})"

        if distance is not None:
            text += f"\nDistance={round(distance.item(), 3)}"

        text += f"\nLabel={int(label)}"

        if predicted is not None:
            text += f"\nPredicted={predicted.item()}({'Same' if predicted.item() == 0 else 'Different'})"

        plt.text(width + fontsize + 10, offset / 2 + offset * i, text,
                 verticalalignment='center',
                 fontsize=fontsize,
                 style='italic', fontweight='bold')

    plt.axis("off")
    plt.imshow(imgs, cmap='gray', interpolation='none')
    plt.show()


# USED
def stats_dataset(dataset, idx_to_class, type):
    print(f"\n{type}")
    tot = 0

    print("{:20s}  | {:3s} | {}".format('Class Name', 'ID', '#samples'))
    for key, value in dataset.items():
        print("{:20s}  | {:3s} | {}".format(idx_to_class[key], str(key), len(value)))
        tot += len(value)
    print(f"Total # of images={tot}")


# USED
def save_datasets(root_folder, test_percentage=0.2, validate_percentage=0.1, output_dir='dataset/', prefix=''):
    # load images from folders
    transform = transforms.Compose([transforms.ToTensor()])
    # transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    training_dataset = torchvision.datasets.ImageFolder(root=root_folder, transform=transform)

    # dictionary with key id of class, and value the name of the class (i.e the folder)
    idx_to_class = {val: key for key, val in training_dataset.class_to_idx.items()}

    start = time.time()
    id_classes = list(idx_to_class.keys())

    # number of classes used for testing
    n_test = int(test_percentage * len(id_classes))  # take % for test

    # shuffle the id_classes
    np.random.shuffle(id_classes)

    # get the test id_classes
    test_idx = id_classes[:n_test]

    # the remaining are the training id_classes
    training_idx = id_classes[n_test:]

    test_data = defaultdict(list)
    training_data = defaultdict(list)
    validation_data = defaultdict(list)

    print("#classes={}  |  #training_classes={}  |  #testing_classes={}".format(len(id_classes), len(training_idx),
                                                                                len(test_idx)))
    print('Start generating dictionaries....')
    # split the data in training and testing
    for (image, idx) in training_dataset:

        if idx in test_idx:
            test_data[idx].append(image)
        else:
            training_data[idx].append(image)

    # update training and create validation with the same classes, but different number of images
    print("{:3s} | {:15s} | {:15s} | {:15s}".format("ID", "#tot_samples", "#train_samples", "#val_samples"))
    for (key, images) in training_data.items():
        n_val = int(validate_percentage * len(images))
        np.random.shuffle(images)
        validation_data[key] = images[:n_val]
        training_data[key] = images[n_val:]
        print("{:3s} | {:15s} | {:15s} | {:15s}".format(str(key), str(len(images)), str(len(images[n_val:])),
                                                        str(len(images[:n_val]))))

    print('Dictionary generated!')
    torch.save(training_data, output_dir + f'training_dataset_{prefix}.pt')
    torch.save(validation_data, output_dir + f'validation_dataset_{prefix}.pt')
    torch.save(test_data, output_dir + f'testing_dataset_{prefix}.pt')
    print("Files Saved")

    print("Finished in {}".format(datetime.timedelta(seconds=int(time.time() - start))))

    print('\n==================== SUMMARY ====================')
    stats_dataset(training_data, idx_to_class, "TRAINING")
    stats_dataset(validation_data, idx_to_class, "VALIDATION")
    stats_dataset(test_data, idx_to_class, "TESTING")

    return training_data, validation_data, test_data


if __name__ == '__main__':
    # PIPELINE
    # 1) resize original images (if needed)
    # resize_images(root_dir='dataset/all')

    # 2) split the dataset and save the objects for later usage

    save_datasets(
        root_folder='dataset/all',
        output_dir='dataset/',
        test_percentage=0.1,
        validate_percentage=0.1,
        prefix='fruits'
    )

    pass

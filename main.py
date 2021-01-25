from network import SiameseNetwork
import torch
from PIL import Image
import numpy as np
import utils
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    checkpoint_path = "models/data20200527_1122_34_251403_threshold0.6_margin1_epoch5.tar"
    net = SiameseNetwork().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint['threshold']
    # margin = checkpoint['margin']

    templates = []
    query_image1 = Image.open('dataset/final_testing_images/img4.jpg')
    query_image_arr1 = np.asarray(query_image1)[..., :3]  # remove alpha channel

    query_image2 = Image.open('dataset/final_testing_images/img3.jpg')
    query_image_arr2 = np.asarray(query_image2)[..., :3]  # remove alpha channel

    query_image3 = Image.open('dataset/final_testing_images/img2.jpg')
    query_image_arr3 = np.asarray(query_image3)[..., :3]

    templates.append(('green', utils.get_patches(query_image_arr1)[6][0]))
    templates.append(('red', utils.get_patches(query_image_arr2)[2][0]))
    templates.append(('blue', utils.get_patches(query_image_arr3)[5][0]))


    target_images = ['dataset/final_testing_images/img1.jpg',
                     'dataset/final_testing_images/img2.jpg',
                     'dataset/final_testing_images/img3.jpg',
                     'dataset/final_testing_images/img4.jpg',
                     ]

    for idx, target_image in enumerate(target_images):
        image = Image.open(target_image)
        image_arr = np.asarray(image)[..., :3]

        patches = utils.get_patches(image_arr)

        for id, (color, template) in enumerate(templates):
            template.save(f'dataset/results/template{id}.png')
            pairs = utils.get_pairs(template, patches)

            predictions = []
            distances = []
            for (img1, (img2, bb)) in pairs:
                img1 = transforms.ToTensor()(img1).unsqueeze_(0)
                img2 = transforms.ToTensor()(img2).unsqueeze_(0)

                prediction, euclidean_distance = net.predict(input1=img1, input2=img2, threshold=threshold)

                predictions.append(prediction)
                distances.append(euclidean_distance)

            utils.draw_result(image, pairs, predictions, distances, color)
        image.show()
        image.save(f'dataset/results/result{idx}.png')


if __name__ == '__main__':
    main()

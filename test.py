import yaml
import numpy as np
from pathlib import Path

import cv2

from nn.mrz_transform_net import MrzTransformNet
from mrz import image_corners_to_homography, MrzTransformDatasetGenerator, denormalize_image_corners


def main():
    weights_path = Path('runs') / '03_15_2022__22_20_52' / 'model' / 'model_4.kpt'

    with open(Path('data') / 'data.yaml', 'r') as stream:
        data_config = yaml.safe_load(stream)

    input_image_size = (data_config['model']['input_image_size']['width'],
                        data_config['model']['input_image_size']['height'])
    mrz_code_image_size = (data_config['model']['mrz_code_image_size']['width'],
                           data_config['model']['mrz_code_image_size']['height'])

    net = MrzTransformNet.load(weights_path, input_image_size, mrz_code_image_size, 'cuda:0')

    generator = MrzTransformDatasetGenerator(data_config['datasets']['coco']['train']['path'],
                                             mode='corner_list',
                                             input_image_size=(data_config['model']['input_image_size']['width'],
                                                               data_config['model']['input_image_size']['height']),
                                             mrz_code_image_size=
                                             (data_config['model']['mrz_code_image_size']['width'],
                                              data_config['model']['mrz_code_image_size']['height']),
                                             max_size=int(data_config['datasets']['coco']['train']['max_size']))

    for image, corner_list in generator:

        matrix = net(image)
        matrix = np.linalg.inv(matrix)
        unwarped_image = cv2.warpPerspective(image, matrix, generator.mrz_code_image_size)

        cv2.imshow('input', image)
        cv2.imshow('unwarped', unwarped_image)
        cv2.waitKey(-1)


if __name__ == '__main__':
    main()
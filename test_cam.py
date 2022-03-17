import yaml
import numpy as np
from pathlib import Path

import cv2

from nn.mrz_transform_net import MrzTransformNet
from mrz import get_affine_fit_image_matrix


def main():
    weights_path = Path('data') / 'model.kpt'

    with open(Path('data') / 'data.yaml', 'r') as stream:
        data_config = yaml.safe_load(stream)

    input_image_size = (data_config['model']['input_image_size']['width'],
                        data_config['model']['input_image_size']['height'])
    mrz_code_image_size = (data_config['model']['mrz_code_image_size']['width'],
                           data_config['model']['mrz_code_image_size']['height'])

    net = MrzTransformNet.load(weights_path, input_image_size, mrz_code_image_size, 'cuda:0')

    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()

        input_transform = get_affine_fit_image_matrix((frame.shape[1], frame.shape[0]), net.input_image_size)
        input_image = cv2.warpAffine(frame, input_transform, net.input_image_size)

        matrix = net(input_image)
        matrix = np.linalg.inv(matrix)
        unwarped_image = cv2.warpPerspective(input_image, matrix, net.mrz_code_image_size)

        cv2.imshow('frame', frame)
        cv2.imshow('input', input_image)
        cv2.imshow('unwarped', unwarped_image)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
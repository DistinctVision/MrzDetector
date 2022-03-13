from typing import Tuple, Union
from pathlib import Path
import numpy as np
import cv2
import yaml

from transform import get_camera_matrix, get_look_transform, plane_slice, get_fit_image_matrix


def generate_mrz_image(shape: Tuple[int, int, int] = (60, 490, 3),
                       border: Union[int, Tuple[int, int]] = (15, 5)) -> np.ndarray:
    if isinstance(border, int):
        border = (border, border)
    image = np.zeros(shape, dtype=np.uint8)
    image.fill(255)
    code_strs = ['PLBRALEE<CHENG<<LIU<<<<<<<<<<<<<<<<<<<<<<',
                 'LA000004<9TWN2002293M1511240<<<<<<<<<<<<<04']
    cv2.putText(image, code_strs[0], (border[0], border[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(image, code_strs[1], (border[0], border[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image


if __name__ == '__main__':
    with open(Path('data') / 'data.yaml', 'r') as stream:
        try:
            data_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    im = generate_mrz_image()

    background = cv2.imread(str(Path(data_config['dataset']['train_directory']) / '000000000139.jpg'))

    camera_matrix = get_camera_matrix((background.shape[1], background.shape[0]), 1.2)
    P = plane_slice(get_look_transform((-70, -30, 20), 500.0))
    T = np.dot(camera_matrix, P)
    F = get_fit_image_matrix((background.shape[1], background.shape[0]), T, (im.shape[1], im.shape[0]))

    T = np.dot(F, T)

    im = cv2.warpPerspective(im, T, (background.shape[1], background.shape[0]),
                             dst=background,  borderMode=cv2.BORDER_TRANSPARENT)

    cv2.imshow('image', background)

    unwarped_image = cv2.warpPerspective(im, np.linalg.inv(T), (490, 60))

    cv2.imshow('original', unwarped_image)

    cv2.waitKey(-1)

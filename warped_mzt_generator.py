from typing import Tuple, Union, Iterable
from pathlib import Path
import os
import random
import numpy as np
import cv2

from transform import get_camera_matrix, get_look_transform, remove_z_axis, get_fit_transformed_image_matrix, \
    get_affine_fit_image_matrix


class WarpedMztGenerator(Iterable):
    def __init__(self,
                 dataset_directory_path: Union[Path, str],
                 input_image_size: Tuple[int, int] = (640, 480),
                 mzt_code_image_size: Tuple[int, int] = (490, 60),
                 limit_angles: Tuple[float, float, float] = (60, 60, 30),
                 focal_length_range: Tuple[float, float] = (0.75, 1.6),
                 distance_range: Tuple[float, float] = (100, 1500),
                 ext_scale_range: Tuple[float, float] = (-0.1, 0.1)):
        dataset_directory_path = Path(dataset_directory_path).absolute()
        assert dataset_directory_path.is_dir()

        self._mzt_code_image_size = mzt_code_image_size[:2]
        self._input_image_size = input_image_size[:2]
        self.limit_angles = limit_angles[:3]
        self.focal_length_range = focal_length_range[:2]
        self.distance_range = distance_range[:2]
        self.ext_scale_range = ext_scale_range

        self._image_paths = []
        for file in os.listdir(str(dataset_directory_path)):
            if file.endswith(".jpg") or file.endswith('.png'):
                self._image_paths.append(dataset_directory_path / file)
        self._iter_index = -1

    @property
    def input_image_size(self) -> Tuple[int, int]:
        return self._input_image_size

    @property
    def mzt_code_image_size(self) -> Tuple[int, int]:
        return self._mzt_code_image_size

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        self._iter_index += 1
        if self._iter_index >= len(self._image_paths):
            raise StopIteration
        return self[self._iter_index]

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        background = cv2.imread(str(self._image_paths[index]))
        fit_matrix = get_affine_fit_image_matrix((background.shape[1], background.shape[0]),
                                                 self._input_image_size)
        background = cv2.warpAffine(background, fit_matrix, self._input_image_size)

        focal_length = random.uniform(self.focal_length_range[0], self.focal_length_range[1])
        angles = (random.uniform(- self.limit_angles[0], self.limit_angles[0]),
                  random.uniform(- self.limit_angles[1], self.limit_angles[1]),
                  random.uniform(- self.limit_angles[2], self.limit_angles[2]))
        distance = random.uniform(self.distance_range[0], self.distance_range[1])
        ext_scale = random.uniform(self.ext_scale_range[0], self.ext_scale_range[1])

        transform_matrix = self._generate_transform_matrix(focal_length, angles, distance, ext_scale)
        mzt_code_image = self._generate_mzt_code_image()
        image = cv2.warpPerspective(mzt_code_image, transform_matrix, self._input_image_size,
                                    dst=background, borderMode=cv2.BORDER_TRANSPARENT)

        return image, transform_matrix

    def _generate_mzt_code_image(self,
                                 border: Union[int, Tuple[int, int]] = (15, 5)) -> np.ndarray:
        if isinstance(border, int):
            border = (border, border)
        image = np.zeros((self._mzt_code_image_size[1], self._mzt_code_image_size[0], 3), dtype=np.uint8)
        image[:, :] = (255, 255, 255)
        code_strs = ['PLBRALEE<CHENG<<LIU<<<<<<<<<<<<<<<<<<<<<<',
                     'LA000004<9TWN2002293M1511240<<<<<<<<<<<<<04']
        cv2.putText(image, code_strs[0], (border[0], border[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(image, code_strs[1], (border[0], border[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return image

    def _generate_transform_matrix(self,
                                   focal_length: float,
                                   angles: Tuple[float, float, float],
                                   distance: float,
                                   ext_scale: float) -> np.ndarray:
        K = get_camera_matrix((self._input_image_size[1], self._input_image_size[0]), focal_length)
        Rt = remove_z_axis(get_look_transform(angles, distance))
        T = np.dot(K, Rt)
        F = get_fit_transformed_image_matrix((self._mzt_code_image_size[0], self._mzt_code_image_size[1]),
                                             (self._input_image_size[0], self._input_image_size[1]), T, ext_scale)
        T = np.dot(F, T)
        return T


if __name__ == '__main__':
    import yaml

    from transform import matrix_to_image_corners, image_corners_to_matrix

    with open(Path('data') / 'data.yaml', 'r') as stream:
        data_config = yaml.safe_load(stream)

    generator = WarpedMztGenerator(data_config['datasets']['coco']['train_directory'])

    for image, matrix in generator:

        matrix /= matrix[2, 2]
        image_corners = matrix_to_image_corners(matrix, generator.mzt_code_image_size)
        matrix1 = image_corners_to_matrix(image_corners, generator.mzt_code_image_size)
        matrix = matrix1

        output_image = cv2.warpPerspective(image, np.linalg.inv(matrix), generator.mzt_code_image_size)

        for i in range(4):
            c_a, c_b = image_corners[i], image_corners[(i + 1) % 4]
            cv2.line(image, (int(c_a[0]), int(c_a[1])), (int(c_b[0]), int(c_b[1])), (0, 255, 0), 2)
        cv2.imshow('input', image)

        cv2.imshow('unwarped', output_image)
        cv2.waitKey(-1)

from typing import Tuple, Union, Iterable, Optional, List
from pathlib import Path
import os
import random
import numpy as np
import cv2

from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

from mrz.transform import get_camera_matrix, get_look_transform, remove_z_axis, get_fit_transformed_image_matrix, \
    get_affine_fit_image_matrix, homography_to_image_corners, normalize_image_corners, denormalize_image_corners
from mrz.mrz_transform_dataset_reader import MrzTransformDatasetReader
from mrz.augmentations import gauss_noise, salt_pepper_noise, speckle_noise, gauss_blur


class MrzTransformDatasetGenerator(Iterable):
    def __init__(self,
                 dataset_directory_path: Union[Path, str],
                 codes_path: Union[Path, str],
                 mode: str = 'matrix',
                 input_image_size: Tuple[int, int] = (640, 480),
                 mrz_code_image_size: Tuple[int, int] = (490, 60),
                 limit_angles: Tuple[float, float, float] = (60, 60, 30),
                 focal_length_range: Tuple[float, float] = (0.75, 1.6),
                 distance_range: Tuple[float, float] = (100, 1500),
                 ext_scale_range: Tuple[float, float] = (-0.1, 0.3),
                 max_size: int = -1,
                 enable_augmentations: bool = True):
        dataset_directory_path = Path(dataset_directory_path).absolute()
        assert dataset_directory_path.is_dir(), f'Invalid path: {dataset_directory_path}'
        assert mode in {'matrix', 'corner_list'}, f'Invalid mode: {mode}'

        self.fonts = [ImageFont.truetype(str(Path('data') / f'font1.ttf'), size=22),
                      ImageFont.truetype(str(Path('data') / f'font2.ttf'), size=19),
                      ImageFont.truetype(str(Path('data') / f'font3.ttf'), size=18),
                      ImageFont.truetype(str(Path('data') / f'font4.ttf'), size=14)]

        self._augmentations = None
        if enable_augmentations:
            self._augmentations = [gauss_noise, salt_pepper_noise, speckle_noise]

        self.mode = mode

        self._mrz_code_image_size = mrz_code_image_size[:2]
        self._input_image_size = input_image_size[:2]
        self.limit_angles = limit_angles[:3]
        self.focal_length_range = focal_length_range[:2]
        self.distance_range = distance_range[:2]
        self.ext_scale_range = ext_scale_range

        self._image_paths = []
        for file in os.listdir(str(dataset_directory_path)):
            if file.endswith(".jpg") or file.endswith('.png'):
                self._image_paths.append(dataset_directory_path / file)

        self._read_codes(codes_path)

        if max_size > 0:
            self._image_paths = self._image_paths[:max_size]
        self._iter_index = -1

    @property
    def input_image_size(self) -> Tuple[int, int]:
        return self._input_image_size

    @property
    def mrz_code_image_size(self) -> Tuple[int, int]:
        return self._mrz_code_image_size

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, List[Tuple[float, float]]]]:
        self._iter_index += 1
        if self._iter_index >= len(self._image_paths):
            raise StopIteration
        return self[self._iter_index]

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[np.ndarray, np.ndarray],
                                               Tuple[np.ndarray, List[Tuple[float, float]]]]:
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

        labels = transform_matrix
        if self.mode == 'corner_list':
            labels = homography_to_image_corners(transform_matrix, self._mrz_code_image_size)
            labels = normalize_image_corners(labels, self.input_image_size)
            label_list = []
            for x, y in labels:
                label_list.extend([x, y])
            labels = label_list

        if self._augmentations:
            blur_size = random.randint(0, 2)
            if blur_size > 0:
                image = gauss_blur(image, blur_size)

            index_f = random.randint(0, len(self._augmentations) - 1)
            image = self._augmentations[index_f](image)

        return image, labels

    def _read_codes(self, codes_path: Union[Path, str]):
        self._codes = []
        with open(codes_path, 'r') as file:
            for line in file.readlines():
                strs = line.strip().split(':')[:2]
                self._codes.append((strs[0], strs[1]))

    def _generate_mzt_code_image(self,
                                 border: Union[int, Tuple[int, int]] = (20, 10)) -> np.ndarray:
        if isinstance(border, int):
            border = (border, border)
        codes_str = random.choice(self._codes)
        codes_str = f'{codes_str[0]}\n{codes_str[1]}'
        fnt = random.choice(self.fonts)
        out = Image.new("RGB", self._mrz_code_image_size, (255, 255, 255))
        draw_context = ImageDraw.Draw(out)
        spacing = random.randint(3, 7)
        draw_context.multiline_text((border[0], border[1]), codes_str, font=fnt, fill=(0, 0, 0), spacing=spacing)
        out = np.array(out)
        # Convert RGB to BGR
        out = out[:, :, ::-1].copy()
        return out

    def _generate_transform_matrix(self,
                                   focal_length: float,
                                   angles: Tuple[float, float, float],
                                   distance: float,
                                   ext_scale: float) -> np.ndarray:
        K = get_camera_matrix((self._input_image_size[1], self._input_image_size[0]), focal_length)
        Rt = remove_z_axis(get_look_transform(angles, distance))
        T = np.dot(K, Rt)
        F = get_fit_transformed_image_matrix((self._mrz_code_image_size[0], self._mrz_code_image_size[1]),
                                             (self._input_image_size[0], self._input_image_size[1]), T, ext_scale)
        T = np.dot(F, T)
        return T


def prepare_dataset(input_dataset_path: Union[Path, str],
                    output_dataset_path: Union[Path, str],
                    input_image_size: Tuple[int, int],
                    mrz_code_image_size: Tuple[int, int],
                    max_size: int = -1,
                    codes_path: Union[Path, str] = Path('data') / 'codes.txt') -> MrzTransformDatasetReader:
    input_dataset_path = Path(input_dataset_path)
    output_dataset_path = Path(output_dataset_path)
    if output_dataset_path.exists():
        return MrzTransformDatasetReader(output_dataset_path, max_size=max_size)
    output_dataset_path.mkdir(parents=True, exist_ok=True)

    generator = MrzTransformDatasetGenerator(input_dataset_path,
                                             codes_path=Path('data') / 'codes.txt',
                                             mode='corner_list',
                                             input_image_size=input_image_size,
                                             mrz_code_image_size=mrz_code_image_size,
                                             max_size=max_size)

    for index, (image, corners) in enumerate(tqdm(generator, f'The dataset preparing: {input_dataset_path}')):
        image_name = f'image_{index}'
        image_path = output_dataset_path / f'{image_name}.jpg'
        labels_path = output_dataset_path / f'{image_name}.txt'
        cv2.imwrite(str(image_path), image)
        with open(labels_path, 'w') as labels_file:
            labels_str = f'{corners[0]} {corners[1]} ' \
                         f'{corners[2]} {corners[3]} ' \
                         f'{corners[4]} {corners[5]} ' \
                         f'{corners[6]} {corners[7]}'
            labels_file.write(labels_str)

    return MrzTransformDatasetReader(output_dataset_path, max_size=max_size)


def main_show():
    import yaml

    from transform import homography_to_image_corners, image_corners_to_homography

    with open(Path('data') / 'data.yaml', 'r') as stream:
        data_config = yaml.safe_load(stream)

    generator = MrzTransformDatasetGenerator(data_config['datasets']['coco']['train']['path'],
                                             Path('data') / 'codes.txt',
                                             mode='corner_list',
                                             input_image_size=(data_config['model']['input_image_size']['width'],
                                                               data_config['model']['input_image_size']['height']),
                                             mrz_code_image_size=
                                             (data_config['model']['mrz_code_image_size']['width'],
                                              data_config['model']['mrz_code_image_size']['height']),
                                             max_size=int(data_config['datasets']['coco']['train']['max_size']))

    for image, corner_list in generator:
        image_corners = [(x, y) for x, y in zip(corner_list[0:8:2], corner_list[1:8:2])]
        image_corners = denormalize_image_corners(image_corners, generator.input_image_size)
        matrix = image_corners_to_homography(image_corners, generator.mrz_code_image_size)

        unwarped_image = cv2.warpPerspective(image, np.linalg.inv(matrix), generator.mrz_code_image_size)

        for i in range(4):
            c_a, c_b = image_corners[i], image_corners[(i + 1) % 4]
            cv2.line(image, (int(c_a[0]), int(c_a[1])), (int(c_b[0]), int(c_b[1])), (0, 255, 0), 2)
        cv2.imshow('input', image)

        cv2.imshow('unwarped', unwarped_image)
        cv2.waitKey(-1)


def main_prepare():
    import yaml
    from transform import image_corners_to_homography

    with open(Path('data') / 'data.yaml', 'r') as stream:
        data_config = yaml.safe_load(stream)

    input_image_size = (data_config['model']['input_image_size']['width'],
                        data_config['model']['input_image_size']['height'])
    mrz_code_image_size = (data_config['model']['mrz_code_image_size']['width'],
                           data_config['model']['mrz_code_image_size']['height'])

    reader = prepare_dataset(data_config['datasets']['coco']['val']['path'],
                             data_config['datasets']['temp_directory'],
                             input_image_size=input_image_size,
                             mrz_code_image_size=mrz_code_image_size,
                             max_size=int(data_config['datasets']['coco']['val']['max_size']))
    for image, corner_list in reader:
        image_corners = [(x, y) for x, y in zip(corner_list[0:8:2], corner_list[1:8:2])]
        image_corners = denormalize_image_corners(image_corners, input_image_size)
        matrix = image_corners_to_homography(image_corners, mrz_code_image_size)

        unwarped_image = cv2.warpPerspective(image, np.linalg.inv(matrix), mrz_code_image_size)

        for i in range(4):
            c_a, c_b = image_corners[i], image_corners[(i + 1) % 4]
            cv2.line(image, (int(c_a[0]), int(c_a[1])), (int(c_b[0]), int(c_b[1])), (0, 255, 0), 2)
        cv2.imshow('input', image)

        cv2.imshow('unwarped', unwarped_image)
        cv2.waitKey(-1)


if __name__ == '__main__':
    # main_prepare()
    main_show()

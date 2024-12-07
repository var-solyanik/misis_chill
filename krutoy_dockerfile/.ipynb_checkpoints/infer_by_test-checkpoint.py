import os
import sys
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torch import nn

# model_path - путь к вашей модели (с именем и расширением файла, относительно скрипта в вашем архиве проекта)
# dataset_path - путь к папке с тестовым датасетом.
# Он состоит из n фотографий c расширением .jpg (гарантируется, что будет только это расширение)
#
# output_path - путь к файлу, в который будут сохраняться результаты (с именем и расширением файла)
dataset_path, output_path = sys.argv[1:]


# TODO ваша работа с моделью
# на вход модели подаются изображения из тестовой выборки
# результатом должен стать json-файл
# В качестве примера здесь показана работа на примере модели из baseline

# Пример функции инференса модели
def add_sobel_as_fourth_channel(image):
    """
    Adds the Sobel operator (edge detection) result as the fourth channel to an input image.

    Parameters:
        image (numpy.ndarray): Input image (H, W, 3) read by cv2.

    Returns:
        numpy.ndarray: Image with the Sobel operator as the fourth channel (H, W, 4).
    """
    if image is None:
        print(image)
        raise ValueError("Input image is None. Please provide a valid image.")
    
    if image.shape[-1] != 3:
        raise ValueError("Input image must have 3 channels (H, W, 3).")

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator (X and Y gradients)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y direction

    # Compute the gradient magnitude
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize to 8-bit (0-255)
    sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_normalized = sobel_normalized.astype(np.uint8)

    # Add Sobel as the fourth channel
    sobel_channel = np.expand_dims(sobel_normalized, axis=-1)  # Shape (H, W, 1)
    image_with_sobel = np.concatenate((image, sobel_channel), axis=-1)  # Shape (H, W, 4)

    return image_with_sobel


def infer_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    image = torch.permute(torch.Tensor(image), (2, 0, 1))
    transform = transforms.Resize((256, 256))
    image = transform(image)

    # Инференс
    return model(image.unsqueeze(0))


# TODO Ваш проект будет перенесен целиком, укажите корректны относительный путь до модели!!!
# TODO Помните, что доступа к интернету не будет и нельзя будет скачать веса модели откуда-то с внешнего ресурса!
model_path = './baseline2.pt'

# Тут показан пример с использованием модели, полученной из бейзлайна
example_model = torch.load('./baseline.pt')
example_model.to('cpu')
example_model.eval()


def create_mask(image_path, results):
    # Загружаем изображение и переводим в градации серого
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Создаем пустую маску с черным фоном
    mask = np.zeros((height, width), dtype=np.uint8)

    # Проходим по результатам и создаем маску
    masks = nn.functional.sigmoid(results).detach().numpy().squeeze().squeeze()
    mask_i_resized = cv2.resize(masks, (width, height), interpolation=cv2.INTER_LINEAR)
    mask[mask_i_resized > 0.5] = 255
    print(mask)
    # for result in results:
    #     masks = 
    #     mask_i = # Получаем маски из результатов
    #     if masks is not None:
    #         for mask_array in masks.data:  # Получаем маски как массивы
    #             mask_i = mask_array.numpy()  # Преобразуем маску в numpy массив
                
    #             # Изменяем размер маски под размер оригинального изображения
    #             mask_i_resized = cv2.resize(mask_i, (width, height), interpolation=cv2.INTER_LINEAR)
                
    #             # Накладываем маску на пустую маску (255 для белого)
    #             mask[mask_i_resized > 0] = 255

    return mask

# Ваша задача - произвести инференс и сохранить маски НЕ в отдельные файлы, а в один файл submit.
# Для этого мы сначала будем накапливать результаты в словаре, а затем сохраним их в JSON.
results_dict = {}

for image_name in os.listdir(dataset_path):
    if image_name.lower().endswith(".jpg"):
        results = infer_image(example_model, os.path.join(dataset_path, image_name))
        mask = create_mask(os.path.join(dataset_path, image_name), results)
        
        # Кодируем маску в PNG в память
        _, encoded_img = cv2.imencode(".png", mask)
        # Кодируем в base64, чтобы поместить в JSON
        encoded_str = base64.b64encode(encoded_img).decode('utf-8')
        results_dict[image_name] = encoded_str

# Сохраняем результаты в один файл "submit" (формат JSON)
submit_path = os.path.join(output_path, "submit.json")
with open(submit_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False)

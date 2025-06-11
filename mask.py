import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def compare_and_merge_masks(mask1_path, mask2_path, output_path):
    # Загрузка масок
    mask1 = io.imread(mask1_path)
    mask2 = io.imread(mask2_path)
    
    # Проверка на идентичность размеров
    if mask1.shape != mask2.shape:
        raise ValueError("Маски должны быть одинакового размера!")
    
    # Находим белые пиксели (значение 255), которые есть в mask1, но отсутствуют в mask2
    white_in_mask1 = (mask1 == 255)
    missing_in_mask2 = (mask2 != 255)
    missing_white = white_in_mask1 & missing_in_mask2
    
    # Создаём новую маску: копируем mask2 и добавляем отсутствующие белые области
    new_mask = mask2.copy()
    new_mask[missing_white] = 255  # Добавляем белые пиксели
    
    # Сохраняем результат
    io.imsave(output_path, new_mask.astype(np.uint8))
    
    # Визуализация (опционально)
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.imshow(mask1, cmap='gray'); plt.title('Маска 1')
    plt.subplot(132); plt.imshow(mask2, cmap='gray'); plt.title('Маска 2')
    plt.subplot(133); plt.imshow(new_mask, cmap='gray'); plt.title('Новая маска')
    plt.show()

# Пример использования
mask1_path = '2_mask.tif'  # Путь к первой маске
mask2_path = '3_mask.tif'  # Путь ко второй маске
output_path = 'merged_mask.tif'  # Путь для сохранения новой маски

compare_and_merge_masks(mask1_path, mask2_path, output_path)

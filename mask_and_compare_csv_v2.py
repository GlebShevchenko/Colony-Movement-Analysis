import numpy as np
import pandas as pd
from skimage import io, measure
import matplotlib.pyplot as plt
import os

def load_csv_data(file_path):
    """Загружает CSV-файл с данными об объектах."""
    df = pd.read_csv(file_path)
    df['xmean_round'] = df['xmean'].round(0).astype(int)
    df['ymean_round'] = df['ymean'].round(0).astype(int)
    return df

def find_complete_objects(mask1, mask2, min_size=100):
    """Находит целые объекты, присутствующие в mask1, но отсутствующие в mask2."""
    # Разница между масками
    diff = (mask1 == 255) & (mask2 != 255)
    
    # Находим связные компоненты
    labeled = measure.label(diff)
    regions = measure.regionprops(labeled, intensity_image=mask1)
    
    complete_objects = []
    new_mask = np.zeros_like(mask1)
    
    for region in regions:
        # Проверяем, что объект достаточно большой и не является фрагментом
        if region.area >= min_size:
            # Координаты ограничивающего прямоугольника
            min_row, min_col, max_row, max_col = region.bbox
            
            # Вырезаем объект из исходной маски
            original_object = mask1[min_row:max_row, min_col:max_col] == 255
            
            # Проверяем, что объект целый (не менее 95% пикселей присутствуют)
            if np.mean(original_object) > 0.50:
                # Добавляем объект в новую маску
                new_mask[labeled == region.label] = 255
                complete_objects.append({
                    'bbox': region.bbox,
                    'centroid': region.centroid,
                    'area': region.area
                })
    
    return new_mask, complete_objects

def match_objects_to_csv(objects, csv_data, max_distance=50):
    """Сопоставляет найденные объекты с данными из CSV."""
    matched = []
    
    for obj in objects:
        y, x = obj['centroid']
        x, y = int(round(x)), int(round(y))
        
        # Ищем ближайший объект в CSV
        distances = np.sqrt((csv_data['xmean_round'] - x)**2 + 
                         (csv_data['ymean_round'] - y)**2)
        closest_idx = distances.idxmin()
        
        if distances[closest_idx] < max_distance:
            matched_obj = csv_data.iloc[closest_idx].to_dict()
            matched_obj['diff_x'] = x
            matched_obj['diff_y'] = y
            matched.append(matched_obj)
    
    return matched

def main():
    # Пути к файлам (замените на свои)
    mask1_path = '2_mask.tif'
    mask2_path = '3_mask.tif'
    csv_path = '2_tab.csv'
    output_csv = 'missing_objects.csv'
    output_mask = 'missing_objects_mask.tif'
    
    # Загрузка данных
    csv_data = load_csv_data(csv_path)
    mask1 = io.imread(mask1_path)
    mask2 = io.imread(mask2_path)
    
    # Поиск целых объектов
    new_mask, complete_objects = find_complete_objects(mask1, mask2)
    
    # Сопоставление с CSV
    matched_objects = match_objects_to_csv(complete_objects, csv_data)
    
    # Сохранение результатов
    io.imsave(output_mask, new_mask)
    
    if matched_objects:
        result_df = pd.DataFrame(matched_objects)
        result_df.to_csv(output_csv, index=False)
        
        print(f"Найдено {len(matched_objects)} отсутствующих объектов:")
        print(f"Результаты сохранены в {output_csv} и {output_mask}")
        
        # Визуализация
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(mask1, cmap='gray')
        plt.title('Исходная маска 1')
        
        plt.subplot(132)
        plt.imshow(mask2, cmap='gray')
        plt.title('Исходная маска 2')
        
        plt.subplot(133)
        plt.imshow(new_mask, cmap='gray')
        for obj in matched_objects:
            plt.plot(obj['diff_x'], obj['diff_y'], 'ro', markersize=5)
        plt.title('Найденные отсутствующие объекты')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Отсутствующие объекты не найдены.")

if __name__ == "__main__":
    main()
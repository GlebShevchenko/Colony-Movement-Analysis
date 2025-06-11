import numpy as np
import pandas as pd
from skimage import io, measure, morphology
import matplotlib.pyplot as plt
import os

def load_csv_data(file_path):
    """Загружает и предобрабатывает CSV с данными об объектах"""
    df = pd.read_csv(file_path)
    df['x_round'] = df['xmean'].round(0).astype(int)
    df['y_round'] = df['ymean'].round(0).astype(int)
    return df

def find_complete_objects(mask1, mask2, min_size=100, completeness_thresh=0.5):
    """
    Находит целые объекты в mask1, отсутствующие в mask2
    с проверкой на полноту объекта и минимальный размер
    """
    diff = (mask1 == 255) & (mask2 != 255)
    labeled = measure.label(diff)
    regions = measure.regionprops(labeled, intensity_image=mask1)
    
    valid_objects = []
    for region in regions:
        if region.area >= min_size:
            # Проверка полноты объекта
            bbox = region.bbox
            obj_slice = mask1[bbox[0]:bbox[2], bbox[1]:bbox[3]] == 255
            completeness = np.sum(region.image) / np.sum(obj_slice)
            
            if completeness >= completeness_thresh:
                valid_objects.append({
                    'coords': region.coords,
                    'centroid': region.centroid,
                    'area': region.area,
                    'bbox': bbox
                })
    
    return valid_objects

def create_combined_mask(base_mask, new_objects, existing_mask):
    """
    Создает новую маску с проверкой на пересечения
    """
    combined_mask = existing_mask.copy()
    collision_mask = np.zeros_like(existing_mask, dtype=bool)
    
    for obj in new_objects:
        # Проверяем пересечения с существующими объектами
        obj_pixels = tuple(obj['coords'].T)
        if np.any(combined_mask[obj_pixels] == 255):
            collision_mask[obj_pixels] = True
        else:
            combined_mask[obj_pixels] = 255
    
    return combined_mask, collision_mask

def match_objects_to_csv(objects, csv_data, max_distance=50):
    """Сопоставляет объекты с данными из CSV"""
    matched = []
    used_indices = set()
    
    for obj in objects:
        y, x = obj['centroid']
        x, y = int(round(x)), int(round(y))
        
        distances = np.sqrt((csv_data['x_round'] - x)**2 + 
                          (csv_data['y_round'] - y)**2)
        
        # Ищем ближайший неиспользованный объект
        for idx in np.argsort(distances):
            if idx not in used_indices and distances[idx] < max_distance:
                matched_obj = csv_data.iloc[idx].to_dict()
                matched_obj.update({
                    'found_x': x,
                    'found_y': y,
                    'distance': distances[idx]
                })
                matched.append(matched_obj)
                used_indices.add(idx)
                break
    
    return matched

def visualize_results(mask1, mask2, combined_mask, collisions, matched_objects):
    """Визуализация результатов"""
    plt.figure(figsize=(18, 6))
    
    plt.subplot(141)
    plt.imshow(mask1, cmap='gray')
    plt.title('Исходная маска 1')
    
    plt.subplot(142)
    plt.imshow(mask2, cmap='gray')
    plt.title('Исходная маска 2')
    
    plt.subplot(143)
    plt.imshow(combined_mask, cmap='gray')
    plt.title('Комбинированная маска')
    
    plt.subplot(144)
    plt.imshow(collisions, cmap='Reds', alpha=0.5)
    plt.imshow(combined_mask, cmap='gray', alpha=0.3)
    for obj in matched_objects:
        plt.plot(obj['found_x'], obj['found_y'], 'bo', markersize=5)
    plt.title('Коллизии и найденные объекты')
    
    plt.tight_layout()
    plt.show()

def main():
    # Параметры
    mask1_path = '2_mask.tif'
    mask2_path = '3_mask.tif' 
    csv_path = '2_tab.csv'
    output_dir = 'results'
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Загрузка данных
    mask1 = io.imread(mask1_path)
    mask2 = io.imread(mask2_path)
    csv_data = load_csv_data(csv_path)
    
    # Поиск объектов для добавления
    objects_to_add = find_complete_objects(mask1, mask2)
    
    # Создание комбинированной маски с проверкой коллизий
    combined_mask, collision_mask = create_combined_mask(mask1, objects_to_add, mask2)
    
    # Сопоставление с CSV
    matched_objects = match_objects_to_csv(objects_to_add, csv_data)
    
    # Сохранение результатов
    io.imsave(os.path.join(output_dir, 'combined_mask.tif'), combined_mask)
    io.imsave(os.path.join(output_dir, 'collision_mask.tif'), collision_mask.astype(np.uint8)*255)
    
    if matched_objects:
        result_df = pd.DataFrame(matched_objects)
        result_df.to_csv(os.path.join(output_dir, 'matched_objects.csv'), index=False)
        
        print(f"Найдено {len(matched_objects)} объектов для добавления")
        print(f"Сохранено в {output_dir}:")
        print("- combined_mask.tif (итоговая маска)")
        print("- collision_mask.tif (маска коллизий)")
        print("- matched_objects.csv (соответствие объектов)")
        
        # Визуализация
        visualize_results(mask1, mask2, combined_mask, collision_mask, matched_objects)
    else:
        print("Объекты для добавления не найдены")

if __name__ == "__main__":
    main()
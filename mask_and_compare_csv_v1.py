import numpy as np
import pandas as pd
from skimage import io, measure
import matplotlib.pyplot as plt

def load_csv_data(file_path):
    """Загружает CSV-файл с данными об объектах."""
    df = pd.read_csv(file_path)
    # Округляем координаты для сравнения
    df['xmean_round'] = df['xmean'].round(0).astype(int)
    df['ymean_round'] = df['ymean'].round(0).astype(int)
    return df

def compare_masks(mask1_path, mask2_path):
    """Сравнивает две маски и возвращает разницу (новые белые области)."""
    mask1 = io.imread(mask1_path)
    mask2 = io.imread(mask2_path)
    
    if mask1.shape != mask2.shape:
        raise ValueError("Маски должны быть одинакового размера!")
    
    # Находим пиксели, которые есть в mask1, но отсутствуют в mask2
    diff_mask = (mask1 == 255) & (mask2 != 255)
    return diff_mask.astype(np.uint8) * 255

def find_objects_in_diff(diff_mask, csv_data):
    """Находит объекты из CSV, соответствующие новым областям в маске."""
    # Находим связные компоненты в разнице масок
    labeled_diff = measure.label(diff_mask == 255)
    regions = measure.regionprops(labeled_diff)
    
    matched_objects = []
    
    for region in regions:
        # Координаты центра новой области
        y, x = region.centroid
        x, y = int(round(x)), int(round(y))
        
        # Ищем ближайший объект в CSV-данных
        distances = np.sqrt((csv_data['xmean_round'] - x)**2 + 
                          (csv_data['ymean_round'] - y)**2)
        closest_idx = distances.idxmin()
        
        if distances[closest_idx] < 50:  # Максимальное допустимое расстояние
            obj = csv_data.iloc[closest_idx]
            matched_objects.append({
                'Number': obj['Number'],
                'x': x,
                'y': y,
                'intensity': obj['channel1_mean'],
                'distance': distances[closest_idx]
            })
    
    return matched_objects

def main():
    # Пути к файлам (замените на ваши)
    mask1_path = '2_mask.tif'
    mask2_path = '3_mask.tif'
    csv_path = '2_tab.csv'
    
    # Загружаем данные
    csv_data = load_csv_data(csv_path)
    diff_mask = compare_masks(mask1_path, mask2_path)
    
    # Находим соответствующие объекты
    matched_objects = find_objects_in_diff(diff_mask, csv_data)
    
    # Сохраняем разницу масок
    io.imsave('difference_mask.tif', diff_mask)
    
    # Вывод результатов
    print("Найдены новые объекты:")
    for obj in matched_objects:
        print(f"Объект {obj['Number']}: Координаты=({obj['x']}, {obj['y']}), "
              f"Интенсивность={obj['intensity']:.2f}, "
              f"Расстояние до центра={obj['distance']:.1f}")

    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(diff_mask, cmap='gray')
    plt.title('Разница между масками')
    
    plt.subplot(122)
    plt.imshow(io.imread(mask1_path), cmap='gray')
    for obj in matched_objects:
        plt.plot(obj['x'], obj['y'], 'ro', markersize=5)
    plt.title('Объекты на исходной маске')
    plt.show()

if __name__ == "__main__":
    main()

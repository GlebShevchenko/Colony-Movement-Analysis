import pandas as pd
import numpy as np

def load_and_preprocess(file_path):
    """Загружает CSV-файл и предобрабатывает данные."""
    df = pd.read_csv(file_path)
    # Округляем координаты для сравнения (чтобы избежать ошибок из-за float)
    df['xmean_round'] = df['xmean'].round(2)
    df['ymean_round'] = df['ymean'].round(2)
    return df

def find_missing_centers(df1, df2):
    """Находит центры из df1, отсутствующие в df2."""
    # Создаем множества координат для быстрого сравнения
    coords_df1 = set(zip(df1['xmean_round'], df1['ymean_round']))
    coords_df2 = set(zip(df2['xmean_round'], df2['ymean_round']))
    
    # Находим разницу: центры из df1, которых нет в df2
    missing_coords = coords_df1 - coords_df2
    
    # Получаем строки из df1, соответствующие отсутствующим центрам
    missing_centers = df1[df1.apply(lambda row: (row['xmean_round'], row['ymean_round']) in missing_coords, axis=1)]
    return missing_centers

def main():
    # Загрузка данных
    df1 = load_and_preprocess('2_tab.csv')  # Первый файл
    df2 = load_and_preprocess('3_tab.csv') # Второй файл
    
    # Поиск отсутствующих центров
    missing_in_df2 = find_missing_centers(df1, df2)
    missing_in_df1 = find_missing_centers(df2, df1)
    
    # Вывод результатов
    print("Центры, присутствующие в _tab.csv, но отсутствующие в 2_tab.csv:")
    if not missing_in_df2.empty:
        for _, row in missing_in_df2.iterrows():
            print(f"Центр {row['Number']}: Интенсивность={row['channel1_mean']:.2f}, Координаты=({row['xmean']:.2f}, {row['ymean']:.2f})")
    else:
        print("Все центры из _tab.csv присутствуют в 2_tab.csv.")
    
    print("\nЦентры, присутствующие в 2_tab.csv, но отсутствующие в _tab.csv:")
    if not missing_in_df1.empty:
        for _, row in missing_in_df1.iterrows():
            print(f"Центр {row['Number']}: Интенсивность={row['channel1_mean']:.2f}, Координаты=({row['xmean']:.2f}, {row['ymean']:.2f}")
    else:
        print("Все центры из 2_tab.csv присутствуют в _tab.csv.")

if __name__ == "__main__":
    main()

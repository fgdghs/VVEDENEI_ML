import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    'video_id': [f'V{i:03d}' for i in range(1, 21)],
    'views_start': np.random.randint(1000, 10000, 20),
    'views_end': np.random.randint(3000, 50000, 20),
    'category': np.random.choice(['Юмор', 'IT', 'Еда'], 20)
}

df = pd.DataFrame(data)


df['growth_rate'] = ((df['views_end'] - df['views_start']) / df['views_start']) * 100

print("=== Все видео с рассчитанным growth_rate ===")
print(df[['video_id', 'category', 'views_start', 'views_end', 'growth_rate']].round(2))

hyped_videos = df.nlargest(3,"views_end")

print("\n=== Топ 3 видео по просмотрам (Хайповые) ===")
print(hyped_videos[['video_id', 'category','views_end' ,'growth_rate']].round(2))

category_count = hyped_videos['category'].value_counts()

print("\n=== Количество хайповых видео по категориям ===")
print(category_count)

top_3 = df.nlargest(3, 'growth_rate')

print("\n=== ТОП-3 самых быстрорастущих видео ===")
print(top_3[['video_id', 'category', 'views_start', 'views_end', 'growth_rate']].round(2))
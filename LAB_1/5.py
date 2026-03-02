import pandas as pd
import numpy as np

np.random.seed(42)

users = [f'User_{i}' for i in range(1, 11)]
genres = ['Рок', 'Поп', 'Техно', 'Рэп']

data = np.random.randint(0, 100, size=(10, 4))

df = pd.DataFrame(data, columns=genres, index=users)

# 1
matrix = df.to_numpy()

# 2
row_sums = matrix.sum(axis=1, keepdims=True) #сумма элеметов для каждой строки
normalized_matrix = matrix / row_sums

print("Нормированная матрица")
print(normalized_matrix)
print("сумма по сторокам")
print(normalized_matrix.sum(axis=1))
print()

# 3
first_user = normalized_matrix[0] 

distances = np.sqrt(np.sum((normalized_matrix - first_user) ** 2, axis=1))

print("3. Евклидовы расстояния от User_1 до всех пользователей ")
for i, (user, dist) in enumerate(zip(users, distances)):
    print(f"   {user}: {dist:.4f}")
print(" ")

# 4
distances_copy = distances.copy()
distances_copy[0] = np.inf  # Исключаем User_1

# Находим индекс пользователя с минимальным расстоянием
most_min_idx = np.argmin(distances_copy)
most_similar_user = users[most_min_idx]
min_distance = distances[most_min_idx]

print("4")
print(f"   Пользователь с наиболее похожими предпочтениями: {most_min_idx}")
print(f"   Минимальное расстояние: {min_distance:.4f}")


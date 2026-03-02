import numpy as np

# 1 и 2
tensor = np.random.normal(loc=0, scale=1, size=(10, 64, 64, 3))

# 3
random_img_index = np.random.randint(0, 10)

tensor[random_img_index, 3::4, :, :] = [255, 0, 0]
print(tensor)

# 4
average_image = np.mean(tensor, axis=0)
print(average_image.mean())

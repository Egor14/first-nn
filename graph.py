from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
from sklearn import preprocessing

image = Image.open("C:\\Users\\Egor1\\Desktop\\lol\\second2.jpg")  # Открываем изображение.
draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
width = image.size[0]  # Определяем ширину.
height = image.size[1]  # Определяем высоту.
pix = image.load()  # Выгружаем значения пикселей.
l = []
X_test = np.array([(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14.9, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 15,
    0, 0, 0, 0, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 14.9, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0)])
for i in range(width):
    for j in range(height):
        l.append(15 - round(pix[j, i][0] / 17))
X_test[0] = l
print(X_test)
# X_scale = StandardScaler()
# X_test = X_scale.fit_transform(X_test)
X_test = preprocessing.scale(X_test)
print(X_test)


# resized_img = image.resize((8, 8), Image.ANTIALIAS)
# print()
# for i in range(8):
#     for j in range(8):
#         print(pix[i, j][0])
#         print(pix[i, j][1])
#         print(pix[i, j][2])

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


def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


def f(x):
    # функция активации
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    # производная
    return f(x) * (1 - f(x))


def weights(nn_structure):
    # присвоение случайных значений весов и смещений
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l - 1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b


def delta_values(nn_structure):
    # присвоение нудевых значений рахнице весов и смещений
    delta_W = {}
    delta_b = {}
    for l in range(1, len(nn_structure)):
        delta_W[l] = np.zeros((nn_structure[l], nn_structure[l - 1]))
        delta_b[l] = np.zeros((nn_structure[l],))
    return delta_W, delta_b


def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l + 1] = W[l].dot(node_in) + b[l]
        h[l + 1] = f(z[l + 1])
    return h, z


def out_delta(y, h_out, z_out):
    # значение дельты на выходном слое
    return -(y - h_out) * f_deriv(z_out)


def hidden_delta(delta_plus_1, w_l, z_l):
    # значение дельты в скрытых слоях
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)


def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    # обучение методом обратного распространения через градиентный спуск
    W, b = weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt % 1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        delta_W, delta_b = delta_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}

            # запуск процесса прямого распространения

            h, z = feed_forward(X[i, :], W, b)

            # обратное распространение

            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = out_delta(y[i, :], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i, :] - h[l]))
                else:
                    if l > 1:
                        delta[l] = hidden_delta(delta[l + 1], W[l], z[l])
                    delta_W[l] += np.dot(delta[l + 1][:, np.newaxis], np.transpose(h[l][:, np.newaxis]))
                    delta_b[l] += delta[l + 1]

        # запуск градиентного спуска для корректировки весов и смещений

        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0 / m * delta_W[l])
            b[l] += -alpha * (1.0 / m * delta_b[l])

        # изменение значения общей оценки

        avg_cost = 1.0 / m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i], W, b)
        y[i] = np.argmax(h[n_layers])
    return y


if __name__ == "__main__":
    # загрузка и разделение данных
    digits = load_digits()
    X_scale = StandardScaler()
    X = X_scale.fit_transform(digits.data)
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


    y_v_train = convert_y_to_vect(y_train)
    y_v_test = convert_y_to_vect(y_test)

    # указываю количество нейронов

    nn_structure = [64, 30, 10]

    # тренировка сети

    W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

    y_pred = predict_y(W, b, X_test, 3)
    print('Prediction accuracy is {}%'.format(accuracy_score(y_test, y_pred) * 100))

    X_test = np.array([(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14.9, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 15,
        0, 0, 0, 0, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0), (0., 0., 5., 13., 9., 1., 0., 0., 0., 0., 13.,
                      15., 10., 15., 5., 0., 0., 3., 15., 2., 0., 11.,
                      8., 0., 0., 4., 12., 0., 0., 8., 8., 0., 0.,
                      5., 8., 0., 0., 9., 8., 0., 0., 4., 11., 0.,
                      1., 12., 7., 0., 0., 2., 14., 5., 10., 12., 0.,
                      0., 0., 0., 6., 13., 10., 0., 0., 0.),
        (0., 0., 0., 12., 13., 5., 0., 0., 0., 0., 0., 11., 16., 9., 0.,
         0., 0., 0.,
         3., 15., 16., 6., 0., 0., 0., 7., 15., 16., 16., 2., 0., 0., 0., 0., 1., 16.,
         16., 3., 0., 0., 0., 0., 1., 16., 16., 6., 0., 0., 0., 0., 1., 16., 16., 6.,
         0., 0., 0., 0., 0., 11., 16., 10., 0., 0.),
        (0., 0., 0., 4., 15., 12., 0., 0., 0., 0., 3., 16., 15., 14., 0., 0., 0.,
         0.,
         8., 13., 8., 16., 0., 0., 0., 0., 1., 6., 15., 11., 0., 0., 0., 1., 8., 13.,
         15., 1., 0., 0., 0., 9., 16., 16., 5., 0., 0., 0., 0., 3., 13., 16., 16., 11.,
         5., 0., 0., 0., 0., 3., 11., 16., 9., 0.)])
    X_scale = StandardScaler()


    print('Нарисовать цифру\n1. Yes\n2. No')
    option = int(input())
    while (option == 1):
        image = Image.open("second2.jpg")  # Открываем изображение.
        draw = ImageDraw.Draw(image)  # Создаем инструмент для рисования.
        width = image.size[0]  # Определяем ширину.
        height = image.size[1]  # Определяем высоту.
        pix = image.load()  # Выгружаем значения пикселей.
        l = []

        for i in range(width):
            for j in range(height):
                l.append(17 - round(pix[j, i][0] / 15))
        X_test[0] = l

        X_test = X_scale.fit_transform(X_test)
        y_pred = predict_y(W, b, X_test, 3)
        print(y_pred[0])
        print('Продолжить?\n1. Yes\n2. No')
        option=int(input())
        X_test = np.array([(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14.9, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 15,
            0, 0, 0, 0, 15, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0), (0., 0., 5., 13., 9., 1., 0., 0., 0., 0., 13.,
                          15., 10., 15., 5., 0., 0., 3., 15., 2., 0., 11.,
                          8., 0., 0., 4., 12., 0., 0., 8., 8., 0., 0.,
                          5., 8., 0., 0., 9., 8., 0., 0., 4., 11., 0.,
                          1., 12., 7., 0., 0., 2., 14., 5., 10., 12., 0.,
                          0., 0., 0., 6., 13., 10., 0., 0., 0.),
            (0., 0., 0., 12., 13., 5., 0., 0., 0., 0., 0., 11., 16., 9., 0.,
             0., 0., 0.,
             3., 15., 16., 6., 0., 0., 0., 7., 15., 16., 16., 2., 0., 0., 0., 0., 1., 16.,
             16., 3., 0., 0., 0., 0., 1., 16., 16., 6., 0., 0., 0., 0., 1., 16., 16., 6.,
             0., 0., 0., 0., 0., 11., 16., 10., 0., 0.),
            (0., 0., 0., 4., 15., 12., 0., 0., 0., 0., 3., 16., 15., 14., 0., 0., 0.,
             0.,
             8., 13., 8., 16., 0., 0., 0., 0., 1., 6., 15., 11., 0., 0., 0., 1., 8., 13.,
             15., 1., 0., 0., 0., 9., 16., 16., 5., 0., 0., 0., 0., 3., 13., 16., 16., 11.,
             5., 0., 0., 0., 0., 3., 11., 16., 9., 0.)])

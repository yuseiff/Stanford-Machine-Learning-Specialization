import math
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = x[i] + b
        cost += (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost
    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(
    x, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient
):
    j_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i < num_iters:
            j_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
    return w, b, j_history, p_history


def prediction(x, w, b):
    y = w * x + b
    return y


# main()
x_train = np.array([1, 2])
y_train = np.array([300, 500])

m = x_train.shape[0]

w_init = 0
b_init = 0

alpha = 1.0e-2
iterations = 10000

w_final, b_final, j_hist, p_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations, compute_cost, compute_gradient
)

# print(w_final, b_final)

for i in range(3):
    i_x = float(input())
    print(f"prediction value of {i_x}= {prediction(i_x,w_final,b_final)}")

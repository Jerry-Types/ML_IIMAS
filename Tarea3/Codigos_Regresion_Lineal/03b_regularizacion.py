# -*- coding: utf-8 -*-
#
# Código de ejemplo de mínimos cuadrados con regularización
#
# Curso de aprendizaje automatizado
# PCIC, UNAM
#
# Gibran Fuentes-Pineda
# Marzo 2017
#
import numpy as np
import scipy.sparse as sp
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# función para expansión en funciones base
def polynomial_expansion(X, degree):
    phi_X = X
    for i in range(degree - 1):
        powerX = np.power(X, i + 2)
        phi_X = np.column_stack((phi_X, powerX))
    return phi_X

# genera los datos (basado en ejemplo de PMTK3 usado en libro de Murphy)
X_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
               11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
y_train = np.array([3.10341864, -0.36949342, -4.20684311, -5.61381536,
                    -4.09482862, -3.23548442, -2.12902581, -7.28955698,
                    -2.8821206, -8.44323436, -5.9491123, -4.77948529,
                    -2.34705651, -3.11360479, 0.58915552, 3.66236699,
                    3.14385906, 11.92541392, 12.06829608, 13.37635698, 14.84302011])
ones_train = np.ones(X_train.shape[0]) # vector de 1s para matriz de diseño

X_valid = np.array([2.2, 2.7, 8.6, 9.5, 10.4, 10.5, 11.4, 14.9, 17.4, 19.3])
y_valid = np.array([-4.87830149, -2.22417664, -4.78937076, -5.39555669,
                    -1.89941084, -4.39873376, -2.74141712, 0.86251019,
                    8.2396395, 13.25506972])
ones_valid = np.ones(X_valid.shape[0]) # vector de 1s para matriz de diseño

# datos en rango de valores para graficar modelos
X_range = np.linspace(0, 20, 10000)
ones_range = np.ones(X_range.shape[0]) # vector de 1s para matriz de diseño

# grafica datos
train_plot = plt.plot(X_train, y_train, 'ro', label = 'Entrenamiento')
valid_plot = plt.plot(X_valid, y_valid, 'bo', label = u'Validación')
plt.legend()
plt.show()

# expande atributo con función base polinomial grado 1
phi_X_train = polynomial_expansion(X_train, 1)
# crea matríz de diseño
phi_X_train = np.column_stack((ones_train, phi_X_train))
# estima parámetros por mínimos cuadrados
# theta_ml = (XT X)-1 XT y
theta_ml = np.dot(np.linalg.inv(np.dot(np.transpose(phi_X_train), phi_X_train)),
                  np.dot(np.transpose(phi_X_train), y_train))
y_train_hat = np.dot(phi_X_train,theta_ml)
# inspecciona la suma de los cuadrados del error (SSE)
# entre las salidas del modelo y las deseadas
sse_train = np.square(y_train - y_train_hat).sum()
# ahora inspecciona el comportamiento del modelo en el conjunto de validación
phi_X_valid = polynomial_expansion(X_valid, 1)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
# calcula error promedio MSE = SSE / N de entrenamiento y validación
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
# predice para rango de valores para graficar modelo
phi_X_range = polynomial_expansion(X_range, 1)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
# grafica modelo
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con función base polinomial grado 2
phi_X_train = polynomial_expansion(X_train, 2)
phi_X_train = np.column_stack((ones_train, phi_X_train))
theta_ml = np.dot(np.linalg.inv(np.dot(np.transpose(phi_X_train), phi_X_train)),
              np.dot(np.transpose(phi_X_train), y_train))
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 2)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 2)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con función base polinomial grado 10
phi_X_train = polynomial_expansion(X_train, 10)
phi_X_train = np.column_stack((ones_train, phi_X_train))
theta_ml = np.dot(np.linalg.inv(np.dot(np.transpose(phi_X_train), phi_X_train)),
              np.dot(np.transpose(phi_X_train), y_train))
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 10)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 10)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con función base polinomial grado 14
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
theta_ml = np.dot(np.linalg.inv(np.dot(np.transpose(phi_X_train), phi_X_train)),
              np.dot(np.transpose(phi_X_train), y_train))
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con función base polinomial grado 20
phi_X_train = polynomial_expansion(X_train, 20)
phi_X_train = np.column_stack((ones_train, phi_X_train))
theta_ml = np.dot(np.linalg.inv(np.dot(np.transpose(phi_X_train), phi_X_train)),
              np.dot(np.transpose(phi_X_train), y_train))
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# solución numéricamente inestable
# mínimos cuadrados se pueden calcular por medio
# de la descomposición QR:
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
q, r = np.linalg.qr(phi_X_train)
# theta_ml = r^-1 q^T y_train
theta_ml = np.dot(np.dot(np.linalg.inv(r), q.T), y_train)
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con función base polinomial grado 20
phi_X_train = polynomial_expansion(X_train, 20)
phi_X_train = np.column_stack((ones_train, phi_X_train))
q, r = np.linalg.qr(phi_X_train)
# theta_ml = r^-1 q^T y_train
theta_ml = np.dot(np.dot(np.linalg.inv(r), q.T), y_train)
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()
# limitemos el rango de y a [-20,20]
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.ylim([-20,20])
plt.show()
print mse_train
print mse_valid

# equivalente a usar función np.linalg.lstsq
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
theta_ml = np.linalg.lstsq(phi_X_train, y_train)[0]
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite para grado 20
phi_X_train = polynomial_expansion(X_train, 20)
phi_X_train = np.column_stack((ones_train, phi_X_train))
theta_ml = np.linalg.lstsq(phi_X_train, y_train)[0]
y_train_hat = np.dot(phi_X_train,theta_ml)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_ml)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_ml)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# equivalente a usar sklearn.linear_model.LinearRegression
phi_X_train = polynomial_expansion(X_train, 14)
clf = LinearRegression()
linreg  = clf.fit(phi_X_train,y_train)
y_train_hat = linreg.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
y_valid_hat = linreg.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
y_range_hat = linreg.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# equivalente a usar sklearn.linear_model.LinearRegression
phi_X_train = polynomial_expansion(X_train, 20)
clf = LinearRegression()
linreg  = clf.fit(phi_X_train,y_train)
y_train_hat = linreg.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
y_valid_hat = linreg.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
y_range_hat = linreg.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# intenta con regularizacion l2
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
lambda_const = 0.01
regularizer = lambda_const * np.identity(phi_X_train.shape[1])
theta_map = np.dot(np.linalg.inv(regularizer + np.dot(np.transpose(phi_X_train), phi_X_train)),
                   np.dot(np.transpose(phi_X_train), y_train))
y_train_hat = np.dot(phi_X_train,theta_map)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_map)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_map)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# solución numéricamente inestable
# intentemos con descomposición QR
# X_tilde = (X; sqrt(lambda) I) y_tilde = (y 0)
# theta_maop = (X_tilde.T X_tilde)-1 X_tilde.T _tilde
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
lambda_const = 0.01
X_tilde = np.concatenate((phi_X_train, np.sqrt(lambda_const) * np.identity(phi_X_train.shape[1])))
y_tilde = np.concatenate((y_train, np.zeros(phi_X_train.shape[1])))
q, r = np.linalg.qr(X_tilde)
theta_map = np.dot(np.dot(np.linalg.inv(r), q.T), y_tilde)
y_train_hat = np.dot(phi_X_train,theta_map)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_map)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_map)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con lambda más pequeño
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
lambda_const = 0.000000000001
X_tilde = np.concatenate((phi_X_train, np.sqrt(lambda_const) * np.identity(phi_X_train.shape[1])))
y_tilde = np.concatenate((y_train, np.zeros(phi_X_train.shape[1])))
q, r = np.linalg.qr(X_tilde)
theta_map = np.dot(np.dot(np.linalg.inv(r), q.T), y_tilde)
y_train_hat = np.dot(phi_X_train,theta_map)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_map)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_map)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con lambda más grande
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
lambda_const = 100.0
X_tilde = np.concatenate((phi_X_train, np.sqrt(lambda_const) * np.identity(phi_X_train.shape[1])))
y_tilde = np.concatenate((y_train, np.zeros(phi_X_train.shape[1])))
q, r = np.linalg.qr(X_tilde)
theta_map = np.dot(np.dot(np.linalg.inv(r), q.T), y_tilde)
y_train_hat = np.dot(phi_X_train,theta_map)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_map)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_map)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con lambda más grande
phi_X_train = polynomial_expansion(X_train, 14)
phi_X_train = np.column_stack((ones_train, phi_X_train))
lambda_const = 1000000.0
X_tilde = np.concatenate((phi_X_train, np.sqrt(lambda_const) * np.identity(phi_X_train.shape[1])))
y_tilde = np.concatenate((y_train, np.zeros(phi_X_train.shape[1])))
q, r = np.linalg.qr(X_tilde)
theta_map = np.dot(np.dot(np.linalg.inv(r), q.T), y_tilde)
y_train_hat = np.dot(phi_X_train,theta_map)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_map)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_map)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite para grado 20
phi_X_train = polynomial_expansion(X_train, 20)
phi_X_train = np.column_stack((ones_train, phi_X_train))
lambda_const = 0.01
X_tilde = np.concatenate((phi_X_train, np.sqrt(lambda_const) * np.identity(phi_X_train.shape[1])))
y_tilde = np.concatenate((y_train, np.zeros(phi_X_train.shape[1])))
q, r = np.linalg.qr(X_tilde)
theta_map = np.dot(np.dot(np.linalg.inv(r), q.T), y_tilde)
y_train_hat = np.dot(phi_X_train,theta_map)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_map)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_map)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con lambda más grande
# repite para grado 20
phi_X_train = polynomial_expansion(X_train, 20)
phi_X_train = np.column_stack((ones_train, phi_X_train))
lambda_const = 100000000000000000.0
X_tilde = np.concatenate((phi_X_train, np.sqrt(lambda_const) * np.identity(phi_X_train.shape[1])))
y_tilde = np.concatenate((y_train, np.zeros(phi_X_train.shape[1])))
q, r = np.linalg.qr(X_tilde)
theta_map = np.dot(np.dot(np.linalg.inv(r), q.T), y_tilde)
y_train_hat = np.dot(phi_X_train,theta_map)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
phi_X_valid = np.column_stack((ones_valid, phi_X_valid))
y_valid_hat = np.dot(phi_X_valid, theta_map)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
phi_X_range = np.column_stack((ones_range, phi_X_range))
y_range_hat = np.dot(phi_X_range, theta_map)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# Equivalente a usar Ridge en sklearn
# El parámetro es alpha y está en una escala diferente
phi_X_train = polynomial_expansion(X_train, 20)
clf = Ridge(alpha=0.00000000000001, copy_X=True, fit_intercept=True, normalize=True)
ridge  = clf.fit(phi_X_train, y_train)
y_train_hat = ridge.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
y_valid_hat = ridge.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
y_range_hat = ridge.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# intenta con regularizacion l1
phi_X_train = polynomial_expansion(X_train, 14)
clf = Lasso(alpha=0.00000001, copy_X=True, fit_intercept=True, normalize=True)
lasso  = clf.fit(phi_X_train, y_train)
y_train_hat = lasso.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
y_valid_hat = lasso.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
y_range_hat = lasso.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con lambda más pequeño
phi_X_train = polynomial_expansion(X_train, 14)
clf = Lasso(alpha=0.000000000000000001, copy_X=True, fit_intercept=True, normalize=True)
lasso  = clf.fit(phi_X_train, y_train)
y_train_hat = lasso.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
y_valid_hat = lasso.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
y_range_hat = lasso.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con lambda más grande
phi_X_train = polynomial_expansion(X_train, 14)
clf = Lasso(alpha=0.01, copy_X=True, fit_intercept=True, normalize=True)
lasso  = clf.fit(phi_X_train, y_train)
y_train_hat = lasso.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
y_valid_hat = lasso.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
y_range_hat = lasso.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite con lambda más grande
phi_X_train = polynomial_expansion(X_train, 14)
clf = Lasso(alpha=1.0, copy_X=True, fit_intercept=True, normalize=True)
lasso  = clf.fit(phi_X_train, y_train)
y_train_hat = lasso.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 14)
y_valid_hat = lasso.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 14)
y_range_hat = lasso.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

# repite para grado 20
phi_X_train = polynomial_expansion(X_train, 20)
clf = Lasso(alpha=0.00000001, copy_X=True, fit_intercept=True, normalize=True)
lasso  = clf.fit(phi_X_train, y_train)
y_train_hat = lasso.predict(phi_X_train)
sse_train = np.square(y_train - y_train_hat).sum()
phi_X_valid = polynomial_expansion(X_valid, 20)
y_valid_hat = lasso.predict(phi_X_valid)
sse_valid = np.square(y_valid - y_valid_hat).sum()
mse_train = sse_train / X_train.size
mse_valid = sse_valid / X_valid.size
phi_X_range = polynomial_expansion(X_range, 20)
y_range_hat = lasso.predict(phi_X_range)
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range_hat, 'b')
plt.show()

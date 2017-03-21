# -*- coding: utf-8 -*-
#
# Código de ejemplo de regresión lineal con mínimos cuadrados
#
# Curso de aprendizaje automatizado
# PCIC, UNAM
#
# Gibran Fuentes-Pineda
# Septiembre 2016
#
import numpy as np
import matplotlib.pyplot as plt

# carga los datos (fuente: http://onlinestatbook.com/2/case_studies/sat.html)
#
# Tarea de regresión: predecir University GPA a partir de otras calificaciones
# Atributos: High school GPA, Math SAT score, Verbal SAT score, Computer Science GPA
# Salida: University GPA
data = np.loadtxt("sat_gpa.csv")

# Asigna cada atributo a un a variable
hs_gpa = data[:,0]
math_sat = data[:,1]
verbal_sat = data[:,2]
comp_gpa = data[:,3]
univ_gpa = data[:,4]

# grafica salida con respecto a cada atributo
# y visualmente identifica qué atributo es mejor para

# plot High School GPA vs University GPA
plt.subplot(2, 2, 1)
plt.plot(hs_gpa, univ_gpa, 'ro')
plt.xlabel('High School GPA')
plt.ylabel('University GPA')

# plot Math SAT vs University GPA
plt.subplot(2, 2, 2)
plt.plot(math_sat, univ_gpa, 'ro')
plt.xlabel('Math SAT')
plt.ylabel('University GPA')

# plot Verbal SAT vs University GPA
plt.subplot(2, 2, 3)
plt.plot(verbal_sat, univ_gpa, 'ro')
plt.xlabel('Verbal SAT')
plt.ylabel('University GPA')

# plot Computer Science GPA vs University GPA
plt.subplot(2, 2, 4)
plt.plot(comp_gpa, univ_gpa, 'ro')
plt.xlabel('Computer Science GPA')
plt.ylabel('University GPA')

plt.show()

# usa sólo el atributo Computer Science GPA como regresor (sólo para el ejemplo)
# divide aleatoriamente la base de datos en entrenamiento (80%) y validación (20%)
permutation = np.random.permutation(comp_gpa.size)
train_set = permutation[:int(comp_gpa.size * 0.8)]
valid_set = permutation[int(comp_gpa.size * 0.8):]

# asigna valores de Computer Science GPA como atributo para las entradas
X_train = comp_gpa[train_set]
X_valid = comp_gpa[valid_set]
# asigna valores de University GPA como salida deseada
y_train = univ_gpa[train_set]
y_valid = univ_gpa[valid_set]

# datos en rango de valores de entrada para graficar modelo
X_range = np.linspace(2.0, 4.0, 10000)
X_range = X_range[:,np.newaxis]

# Suponemos modelo de salida lineal con sólo un atributo
# elige un par de pesos arbitrariamente
theta0 = 1
theta1 = 4
# calcula salida del modelo con los pesos elegidos
y_train_hat = theta0 + theta1 * X_train

# inspecciona la suma de los cuadrados del error (SSE)
# entre las salidas del modelo y las deseadas
sse_train = (np.square(y_train - y_train_hat).sum())

# ahora inspecciona los datos de validación
y_valid_hat = theta0 + theta1 * X_valid
sse_valid = (np.square(y_valid - y_valid_hat).sum())

# calcula error promedio MSE = SSE / N de entrenamiento y validación
mse_train = sse_train / train_set.size
mse_valid = sse_valid / valid_set.size

# calcula salida para rango de valores de entrada
y_range = theta0 + theta1 * X_range

# grafica salidas del modelo y deseadas para ver ajuste
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range, 'b')
plt.show()

# elige otro par de pesos y repite lo anterior
theta0 = -2
theta1 = 2
y_train_hat = theta0 + theta1 * X_train
# ¿qué error es mayor?
sse_train = (np.square(y_train - y_train_hat).sum())

# ahora inspecciona los datos de validación
y_valid_hat = theta0 + theta1 * X_valid
sse_valid = (np.square(y_valid - y_valid_hat).sum())

# calcula error promedio MSE = SSE / N de entrenamiento y validación
mse_train = sse_train / train_set.size
mse_valid = sse_valid / valid_set.size

# calcula salida para rango de valores de entrada
y_range = theta0 + theta1 * X_range

# grafica salidas del modelo y deseadas para ver ajuste
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range, 'b')
plt.show()

# crea matriz de diseño (agrega 1 a las entradas para la ordenada al origen)
ones = np.ones((X_train.shape[0],1))
X_train_ones = np.concatenate((ones, X_train[:,np.newaxis]), axis=1)

# Para encontrar pesos por mínimos cuadrados, theta_ml = (XT X)-1 XT y
# (XT X)
XTX = np.dot(X_train_ones.T, X_train_ones)
# (XT X)-1
XTX_inv = np.linalg.inv(XTX)
# XT y
XTy = np.dot(X_train_ones.T, y_train)
# theta_ml = (XT X)-1 XT y
theta_ml = np.dot(XTX_inv, XTy)

# calcula la salida y el error con los pesos encontrados
y_train_hat = theta_ml[0] + theta_ml[1] * X_train
# equivalente a y_hat = np.dot(X_train_ones, theta_ml)
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))

# ahora inspecciona los datos de validación
y_valid_hat = theta0 + theta1 * X_valid
sse_valid = (np.square(y_valid - y_valid_hat).sum())

# calcula error promedio MSE = SSE / N de entrenamiento y validación
mse_train = sse_train / train_set.size
mse_valid = sse_valid / valid_set.size

# grafica función ajustada y datos de entrenamiento
# verifica qué tan bien se ajusta la función a los datos
# calcula salida para rango de valores de entrada
y_range = theta_ml[0] + theta_ml[1] * X_range

# grafica salidas del modelo y deseadas para ver ajuste
plt.plot(X_train, y_train, 'ro')
plt.plot(X_range, y_range, 'b')
plt.show()

# verifica qué pasa con la suma de los cuadrados del error (SSE) de entrenamiento
# si asignamos valores cercanos a los parámetros encontrados por mínimos cuadrados
# (donde debe ser mínimo)
y_train_hat = theta_ml[0] + (theta_ml[1] - 0.01) * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))
y_train_hat = theta_ml[0] + (theta_ml[1] + 0.01) * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))
y_train_hat = (theta_ml[0] - 0.01) + theta_ml[1] * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))
y_train_hat = (theta_ml[0] + 0.01) + theta_ml[1] * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))
y_train_hat = (theta_ml[0] - 0.01) + (theta_ml[1] - 0.01) * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))
y_train_hat = (theta_ml[0] + 0.01) + (theta_ml[1] - 0.01) * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))
y_train_hat = (theta_ml[0] - 0.01) + (theta_ml[1] + 0.01) * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))
y_train_hat = (theta_ml[0] + 0.01) + (theta_ml[1] + 0.01) * X_train
sse_train = np.dot((y_train - y_train_hat).T, (y_train - y_train_hat))

import pandas as pd
import seaborn as sns
import numpy as np

def logistica(z):
    return 1 / (1 + np.exp(-z))

# calcula el gradiente 
def gradiente(X, y, p):
    return np.dot(X.T, p - y) / y.size

# realiza descenso del gradiente para regresión logística
def descenso_gradiente(X, y, umbral_convergencia = 0.001, alpha = 0.1):
    # inicializa los parámetros de forma aleatoria
    theta_anterior = np.zeros(X.shape[1])
    theta_nuevo = np.random.rand(X.shape[1])

    # calcula cambio entre parámetros nuevos y anteriores (distancia euclideana)
    distancia_theta = np.linalg.norm(theta_nuevo - theta_anterior)
    
    it = 0
    while distancia_theta > umbral_convergencia:
        # parámetros nuevos se vuelven los anteriores
        theta_anterior = theta_nuevo

        # calcula la probabilidad de que datos sean de la clase 1 con los parámetros anteriores
        p = logistica(np.dot(X, theta_anterior)) # p = logistica(X theta)

        # actualiza parámetros en la dirección del gradiente
        theta_nuevo = theta_anterior - alpha * gradiente(X, y, p)

        # calcula cambio entre parámetros nuevos y anteriores (distancia euclideana)
        distancia_theta = np.linalg.norm(theta_nuevo - theta_anterior)

        it = it + 1
        print "Iteración", it, "theta_nuevo =", theta_nuevo,\
              "Cambio =", distancia_theta

    return theta_nuevo


df_spam =  pd.read_csv("nb_data/spam.csv",delim_whitespace=True,header=None)
#df_spam.shap

X = df_spam.as_matrix()

perm = np.random.permutation(X.shape[0])
X= X[perm,:]

X_train =  X[:int(X.shape[0] * 0.8),:]
X_test = X[int(X.shape[0] * 0.8):,:]

y_train= X_train[:,2000:2001]
y_train = y_train.reshape(4137,)
X_train= X_train[:,0:2000]

y_test= X_test[:,2000:2001]
X_test= X_test[:,0:2000]


theta_emv = descenso_gradiente(X_train, y_train) 
print "theta_emv = ", theta_emv.Tt 


print ("Se ca a caluclar el valor para MSE")

y_entrenamiento_predicha = logistica(np.dot(X_train, theta_emv))
print "MSE entrenamiento = ", np.square(y_entrenamiento - y_entrenamiento_predicha).mean()
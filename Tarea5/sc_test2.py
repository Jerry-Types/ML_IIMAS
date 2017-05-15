# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os
import sys
import io
import scipy as sp
from collections import Counter
import tqdm
from sklearn.externals import joblib


def generate_bag_of_features(inputpath, mkm):
    with ZipFile(inputpath) as imagedb:
        JA_train = []
        A_train = []
        IA_train = [0]
        y_train = []
        JA_test = []
        A_test= []
        IA_test = [0]
        y_test = []
        for entry in tqdm.tqdm(imagedb.infolist()):
            filedir,filename = os.path.split(entry.filename)
            if filename.endswith('.npy'):
                with io.BufferedReader(imagedb.open(entry)) as file:
                    sift = np.load(file)
                    bof = Counter(np.sort(mkm.predict(sift)))
                    restpath,label= os.path.split(os.path.dirname(filedir))
                    if entry.filename.startswith('data_tarea/train/'):
                        JA_train.extend(bof.keys())
                        A_train.extend(bof.values())
                        IA_train.append(IA_train[len(IA_train) - 1] + len(bof))
                        y_train.append(label)
                    else:
                        JA_test.extend(bof.keys())
                        A_test.extend(bof.values())
                        IA_test.append(IA_test[len(IA_test) - 1] + len(bof))
                        y_test.append(label)
                        
    JA_train = np.asarray(JA_train, dtype=np.intc)
    IA_train = np.asarray(IA_train, dtype=np.intc)
    A_train = np.asarray(A_train, dtype=np.intc)

    X_train = sp.sparse.csr_matrix((A_train, JA_train, IA_train),
                                   shape=(len(IA_train) - 1, mkm.cluster_centers_.shape[0]))
    X_train.sort_indices()
    y_train = np.array(y_train)
    
    JA_test = np.asarray(JA_test, dtype=np.intc)
    IA_test = np.asarray(IA_test, dtype=np.intc)
    A_test = np.asarray(A_test, dtype=np.intc)
    X_test = sp.sparse.csr_matrix((A_test, JA_test, IA_test),
                           shape=(len(IA_test) - 1, mkm.cluster_centers_.shape[0]))
    X_test.sort_indices()
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def main(vocabpath="/home/nemis/ML_IIMAS/Tarea5/vocabulario.pkl", inputpath="/home/nemis/ML_IIMAS/Tarea5/imagedb.zip"):
    if os.path.isfile(vocabpath):
        mkm = joblib.load(vocabpath)

        if os.path.isfile(inputpath):
            bdc = generate_bag_of_features(inputpath, mkm)
            joblib.dump(bdc, 'bolsas_caracteristicas.pkl')
        else:
            print "Archivo de base de datos de imágenes no existe"
    else:
        print "Archivo de vocabulario no existe"


main()

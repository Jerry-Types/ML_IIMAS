import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os
import sys
import io
import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

def construct_vocabulary_sift(inputpath, number_of_iterations,pca_sift):
    mkm = MiniBatchKMeans(n_clusters=1000)
    print "Estoy Aqui"
    with ZipFile(inputpath) as imagedb:
        for i in tqdm.trange(number_of_iterations):
            for entry in imagedb.infolist():
                if entry.filename.startswith('data_tarea/train/') and entry.filename.endswith('.npy'):
                    #print (entry.filename)
                    with io.BufferedReader(imagedb.open(entry)) as file:
                        X = np.load(file)
                        #print "Transform"
                        X=pca_sift.transform(X)
                        mkm.partial_fit(X)

    return mkm

def main():
    name_model = "vocabulario_pca_sif.pkl"
    name_zip = "imagedb.zip"
    if os.path.isfile(name_model):
        #print "The file exist"
        pca_sift = joblib.load('pca_sift.pkl')
        mkm = construct_vocabulary_sift(name_zip,100,pca_sift)
        joblib.dump(mkm, 'vocabulario_pca_sift.pkl') 
    else:
        print "File doesn't exists"
        
main()

import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
import os
import sys
import io
import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

def construct_vocabulary_sift(inputpath, number_of_iterations=100):
    mkm = MiniBatchKMeans(n_clusters=1000)
    print "Estoy Aqui"
    with ZipFile(inputpath) as imagedb:
        for i in tqdm.trange(number_of_iterations):
            for entry in imagedb.infolist():
                if entry.filename.startswith('data_tarea/train/') and entry.filename.endswith('.npy'):
                    print (entry.filename)
                    with io.BufferedReader(imagedb.open(entry)) as file:
                        X = np.load(file)
                        mkm.partial_fit(X)

    return mkm

def main(inputpath="/home/nemis/ML_IIMAS/Tarea5/"):
    name_model = "vocabulario.pkl"
    name_zip = "imagedb.zip"
    if os.path.isfile(inputpath+name_model):
        #print "The file exist"
        mkm = construct_vocabulary_sift(inputpath+name_zip)
        joblib.dump(mkm, 'vocabulario.pkl') 
    else:
        print "File doesn't exists"
        
main()

#!/usr/bin/python
from zipfile import ZipFile
import os
import sys
import io
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from collections import Counter
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


def pca_apply_sift(inputpath,n_comp,name_pca):
    """
    This function apply Partial PCA to SIFT vectors.
     
    Args:
        inputpath (str): The name of the zip file that contains the data.
        n_comp (int): The number of components for PCA.

    Returns:
       The function create a pkl file that contains the PCA.
    """
    print ("===The PCA process has started===")
    print (".....")
    ipca_p = IncrementalPCA(n_components=n_comp)
    with ZipFile(inputpath) as imagedb:
        for entry in (imagedb.infolist()):
                if entry.filename.startswith('data_tarea/train/') and entry.filename.endswith('.npy'):
                    #print (entry.filename)
                    with io.BufferedReader(imagedb.open(entry)) as file:
                        X = np.load(file)
                        ipca_p.partial_fit(X)
    joblib.dump(ipca_p, name_pca)
    print ("===The PCA process is over===")


def construct_vocabulary_sift(inputpath, number_of_iterations,name_vocabulary,pca_sift_flag,name_pca_sift):
    if pca_sift_flag == True:
        pca_sift = joblib.load(name_pca_sift)
        print "Load ... pca_sift"
    mkm = MiniBatchKMeans(n_clusters=1000)
    print "Estoy Aqui"
    with ZipFile(inputpath) as imagedb:
        for i in tqdm.trange(number_of_iterations):
            for entry in imagedb.infolist():
                if entry.filename.startswith('data_tarea/train/') and entry.filename.endswith('.npy'):
                    #print (entry.filename)
                    with io.BufferedReader(imagedb.open(entry)) as file:
                        if pca_sift_flag == True:
                            #print "Aplicando PCA"
                            X = np.load(file)
                            X = pca_sift.transform(X)
                            mkm.partial_fit(X)
                        else:
                            X = np.load(file)
                            mkm.partial_fit(X)
    joblib.dump(mkm, name_vocabulary) 
    #return mkm


    
def generate_bag_of_features(inputpath, mkm_name,name_bag,pca_sift_flag,name_pca_sift,flag_apply_pca_histogram):
    mkm = joblib.load(mkm_name)
    if pca_sift_flag == True:
        pca_sift = joblib.load(name_pca_sift)
        print "Load ... pca_sift"
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
                    if pca_sift_flag == True:
                        sift = pca_sift.transform(sift)
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
    if flag_apply_pca_histogram==True:
        svd = TruncatedSVD(n_components=500)
        svd.fit(X_train)
        X_train = svd.transform(X_train)
        X_test = svd.transform(X_test)
        print "Applied SVD to Histograms"
    bdc= X_train,y_train,X_test,y_test
    joblib.dump(bdc, name_bag)

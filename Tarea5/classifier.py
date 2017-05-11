import random
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import tqdm

def batches(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


data_bf = joblib.load("bolsas_caracteristicas.pkl", mmap_mode='c')
X_train = data_bf[0]
y_train = data_bf[1]
X_test = data_bf[2]
y_test = data_bf[3]

print X_train.shape
print X_train[0:2].shape
bs=range(1,1500+1,50)

clf = MultinomialNB()
#print ("The shapes")
#print (X_train[0:10].shape,y_train[0:10].shape)
#clf.partial_fit(X_train[0:10],y_train[0:10],np.unique(y_train))
clf.fit(X_train,y_train)
#print y_train[1]
#for i in range(len(bs)-1):
    #print X_train[bs[i]:bs[i+1]].shape,y_train[bs[i]:bs[i+1]].shape
    #print "Jajaj"
 #   print i
  #  clf.partial_fit(X_train[bs[i]:bs[i+1]],y_train[bs[i]:bs[i+1]],np.unique(y_train))
    #break
#clf = MultinomialNB()
#clf.partial_fit(X_train,y_train,np.unique(y_train))
print clf.predict(X_train)

"""
shuffledRange = range((X_train.shape[0]))
n_iter = 10
for n in range(n_iter):
    random.shuffle(shuffledRange)
    shuffledX = [X_train[i] for i in shuffledRange]
    shuffledY = [y_train[i] for i in shuffledRange]
    for batch in batches(range(len(shuffledX)), 10000):
        clf.partial_fit(shuffledX[batch[0]:batch[-1]+1], shuffledY[batch[0]:batch[-1]+1], np.unique(y_train))
"""

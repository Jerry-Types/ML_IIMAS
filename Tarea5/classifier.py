#!/user/bin/python

import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import tqdm



data_bf = joblib.load("bolsas_caracteristicas_sin_pca.pkl", mmap_mode='c')
X_train = data_bf[0]
y_train = data_bf[1]
X_test = data_bf[2]
y_test = data_bf[3]

print "###Multinomial###"
print "===Multinomial Train==="
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_hat = clf.predict(X_train)
#print y_hat
print accuracy_score(y_train,y_hat)

print "===Multinomial Test==="
y_hat_test= clf.predict(X_test)
print accuracy_score(y_test,y_hat_test)



print "###SGD###"
from sklearn.linear_model import SGDClassifier
print "===SGD Train==="
clf_svm = SGDClassifier(loss="hinge", penalty="l2")
clf_svm.fit(X_train, y_train)
y_hat = clf_svm.predict(X_train)
print accuracy_score(y_train,y_hat)
print "===SGD Test==="
y_hat2= clf_svm.predict(X_test)
#print y_hat2
print accuracy_score(y_test,y_hat2)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#cnf_matrix = confusion_matrix(y_train, y_hat)
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=np.unique(y_train),
#                      title='Confusion matrix, without normalization')
#plt.show()

#we do imports , we need numpy pandas sklearn pyasn installed
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import pyasn
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#we first read our csv using pandas
data=pd.read_csv('reduced.csv',header=None)

#we extract numerical data
X_numerical=data[[2,3,9,11,1]]

#we extract categorical data using dummies
X_categorical=pd.get_dummies(data[[5]])
Y_labled=pd.get_dummies(data[[12]])

#we are interrested only with the dos column
Y_labled=Y_labled[['12_dos']]

#casting to numpy arrays and combining matrix

Y=np.array(Y_labled)
X=np.hstack((np.array(X_categorical),np.array(X_numerical)))
X=np.hstack((X ,  np.array(data[[7]])))

# reducing
#depending on your computing power , you may want to run this cell twice
a, X, b, Y = train_test_split(X, Y, test_size = 0.10, random_state = 0)

aspf=X[:,-1]
print(aspf)
#dimension
row_size=aspf.size
#number of caracteres
nb_attr=len(aspf[0])


#we want to transform the flag to more logical form , using one hot encoding
ASPF=np.ones((row_size,nb_attr))
for i in range(0,row_size):
    for j in range(0,nb_attr):
        if aspf[i][j]=='.':
            ASPF[i,j]=0

X=X[:,:-1]
X=np.hstack((X,ASPF))


#here we want to transform IP adresses to ASN
IP=np.zeros((row_size,2))
asndb = pyasn.pyasn('ipasn_20140513.dat')


for i in range(0,row_size):
    ASN=asndb.lookup(X[i,1])[0]
    if str(ASN)=='None' :
        ASN=0
    IP[i,0]=ASN
    ASN=asndb.lookup(X[i,2])[0]
    if str(ASN)=='None' :
        ASN=0
    IP[i,1]=ASN


X=np.hstack((X[:,0:1],X[:,3:]))
X=np.hstack((X,IP))


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting kneighbor to the Training set
classifier = KNeighborsClassifier(1, n_jobs=-1)
classifier.fit(X_train, y_train)

"""
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


print("the confusion matrix on the test set")
print(cm)


#now, we will test our model on a different part of the dataset
#we have to preprocess it the same way

print("now we will use our trained model on the another sub-dataset")
data2=pd.read_csv('reduced3.csv',header=None)


X_numerical2=data2[[2,3,9,11,1]]

#dummies
X_categorical2=pd.get_dummies(data2[[5]])
Y_labled2=pd.get_dummies(data2[[12]])


#on garde juse dos et normal
Y_labled2=Y_labled2[['12_dos']]


#casting to numpy arrays and combining matrix

Y2=np.array(Y_labled2)
X2=np.hstack((np.array(X_categorical2),np.array(X_numerical2)))
X2=np.hstack((X2 ,  np.array(data2[[7]])))


# reducing
a, X2, b, Y2 = train_test_split(X2, Y2, test_size = 0.10, random_state = 0)

aspf2=X2[:,-1]
print(aspf2)

#dimension
row_size2=aspf2.size
#number of caracteres
nb_attr2=len(aspf2[0])





ASPF2=np.ones((row_size2,nb_attr2))
for i in range(0,row_size2):
    for j in range(0,nb_attr2):
        if aspf2[i][j]=='.':
            ASPF2[i,j]=0



X2=X2[:,:-1]
X2=np.hstack((X2,ASPF2))


#for ip
IP2=np.zeros((row_size2,2))
asndb2 = pyasn.pyasn('ipasn_20140513.dat')


for i in range(0,row_size2):
    ASN2=asndb2.lookup(X2[i,1])[0]
    if str(ASN2)=='None' :
        ASN2=0
    IP2[i,0]=ASN2
    ASN2=asndb2.lookup(X2[i,2])[0]
    if str(ASN2)=='None' :
        ASN2=0
    IP2[i,1]=ASN2


X2=np.hstack((X2[:,0:1],X2[:,3:]))
X2=np.hstack((X2,IP2))


# Feature Scaling
X2 = sc.fit_transform(X2)


# Predicting the Test set results
y_pred2 = classifier.predict(X2)


#print the confusion matrix
cm2 = confusion_matrix(Y2, y_pred2)
print("the confusion matrix on another part of the dataset")
print(cm2)

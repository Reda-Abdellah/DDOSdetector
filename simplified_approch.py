import numpy as np
import pandas as pd

data=pd.read_csv('ddos1range32M.csv',header=None)

X_numerical=data[[8,9,10,11]]

#dummies
X_categorical=pd.get_dummies(data[[4,5,6]])
Y_labled=pd.get_dummies(data[[12]])

#on garde juse dos et normal
Y_labled=Y_labled[['12_dos']]

#casting to numpy arrays and combining matrix

Y=np.array(Y_labled)
X=np.hstack((np.array(X_categorical),np.array(X_numerical)))
aspf=np.array(data[[7]])

#dimension 
row_size=aspf.size
#number of caracteres
nb_attr=len(aspf[0,0])


ASPF=np.ones((row_size,nb_attr))
for i in range(0,row_size):
    for j in range(0,nb_attr):
        if aspf[i,0][j]=='.':
            ASPF[i,j]=0


X=np.hstack((X,ASPF))


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


accuracy=(cm[1,1]+cm[0,0])/(cm[1,1]+cm[0,0]+cm[0,1]+cm[1,0])
print("accuracy is : "+str(accuracy*100)+'%')



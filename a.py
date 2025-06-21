import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data/fever.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())
dataset = data.copy()
X = dataset.drop(['target'], axis = 1)
print(X.head())
y = dataset['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,pred)))




#print(pred[:10])
input=(98,0,1,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems to have fever")
else:
  print("The patient seems to be Normal")

print('Random forset')
rf=accuracy_score(y_test,pred)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
print('logistic classifier')
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,pred)))

#print(pred[:10])
input=(98,0,1,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=classifier.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems have fever")
else:
  print("The patient seems to be Normal")

lr=accuracy_score(y_test,pred)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

#print(pred[:10])
from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,pred)))

input=(102,0,1,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=classifier.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems to have fever")
else:
  print("The patient seems to be Normal")

print('svm')

sv=accuracy_score(y_test,pred)
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,pred)))




#print(pred[:10])
input=(98,0,1,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems to have fever")
else:
  print("The patient seems to be Normal")

print('Random forset')
dt=accuracy_score(y_test,pred)


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print('\n'+'-'*20+'Accuracy Score on the Test set'+'-'*20)                             
print("{:.0%}".format(accuracy_score(y_test,pred)))




#print(pred[:10])
input=(98,0,1,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=model.predict(input_reshaped)
print(pre1)
if(pre1==1): 
  print("The patient seems to have fever")
else:
  print("The patient seems to be Normal")

print('Random forset')
knn=accuracy_score(y_test,pred)


x=[rf,lr,sv,dt,knn]
y=['RF','LR','SVM','DT','KNN']
plt.bar(y,x)
plt.xlabel("Accuracy")
plt.ylabel("Alogritham")

plt.title("Comprasion graph")
plt.show()



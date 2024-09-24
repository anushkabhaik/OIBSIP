import numpy as np
import pandas as pd

iris_data=pd.read_csv('iris.csv')
print(iris_data.head())

X = iris_data.drop(columns=['Species'])
y = iris_data['Species']

from sklearn.preprocessing import LabelEncoder
print("before encoding")
print(y)
l_e = LabelEncoder()
y = l_e.fit_transform(y)
print("after encoding")
print(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100:.2f,"%")


print("Classification Report : \n",classification_report(y_test,y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test,y_pred))


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(iris_data,hue="Species")
plt.show()






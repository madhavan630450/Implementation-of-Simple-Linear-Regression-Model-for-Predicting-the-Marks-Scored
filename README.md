# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and split it into features (X) and target (Y).
2. Split the data into training and testing sets using train_test_split.
3. Train a Linear Regression model on the training set.
4. Evaluate the model using predictions on the test set and plot the results.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MARIMUTHU MATHAVAN
RegisterNumber:  212224230153
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
print(X.shape)
print(Y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE =',mae)
rmse=mean_absolute_error(Y_test,Y_pred)
print('RMSE =',rmse)
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,Y_pred,color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
a=np.array([[13]])
ans=reg.predict(a)
print(ans)
```

## Output:
![Screenshot 2025-03-15 230652](https://github.com/user-attachments/assets/1876e17a-164c-4b03-a77b-6f940d8875db)
![Screenshot 2025-03-15 230808](https://github.com/user-attachments/assets/4bd98eff-2497-43db-830e-760321a1eb4b)
![Screenshot 2025-03-15 230901](https://github.com/user-attachments/assets/6f8ce2ac-5088-4fd9-9fbe-8875b41709a3)
![Screenshot 2025-03-15 230948](https://github.com/user-attachments/assets/28a1a06b-1e3b-413b-ab46-fdbc6bf74253)
![Screenshot 2025-03-15 231021](https://github.com/user-attachments/assets/c3e1876e-f001-4483-8d65-8e3042b819c7)
![download](https://github.com/user-attachments/assets/10b760e5-478b-481e-a644-9d9da7176265)
![Screenshot 2025-03-15 231141](https://github.com/user-attachments/assets/1e5d2d70-bfa9-4cb9-894f-e0f31cc6aaa5)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

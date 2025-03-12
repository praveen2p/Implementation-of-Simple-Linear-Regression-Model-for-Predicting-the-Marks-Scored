##  EX 02-Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAVEEN.K
RegisterNumber: 212223040152

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
  
*/
```

## Output:
## Head values:
![Screenshot 2025-03-11 200704](https://github.com/user-attachments/assets/a2d33411-334e-4728-b878-ce60843dad3d)

## Tail values:
![Screenshot 2025-03-11 200712](https://github.com/user-attachments/assets/c224a324-f4b4-48dd-8faa-cc54ba2b2169)

## Compare Dataset:
![Screenshot 2025-03-11 200730](https://github.com/user-attachments/assets/d9ce4440-606d-4de9-b337-6f834aea4c82)

## Predication values of X and Y:
![Screenshot 2025-03-11 200739](https://github.com/user-attachments/assets/555b8e50-7f41-49c4-a3fe-d3550c0f062c)

## Training set:
![Screenshot 2025-03-11 200754](https://github.com/user-attachments/assets/6121fae4-b9ce-48cd-bb69-f8401458ad66)

## Testing set:

![Screenshot 2025-03-11 200804](https://github.com/user-attachments/assets/28d9a6e9-dfed-4e30-a620-cc1406ea9cbb)

## MSE,MAE and RMSE :

![Screenshot 2025-03-11 200816](https://github.com/user-attachments/assets/e662c8d2-135a-4552-bc9f-bcba24dc2904)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

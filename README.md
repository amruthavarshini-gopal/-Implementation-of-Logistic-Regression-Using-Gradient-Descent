# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## Aim:

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

6.Define a function to predict the Regression value.

7.Print the results.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013  
*/
import pandas as pd
import numpy as np
df=pd.read_csv("Placement_Data.csv")
df
df=df.drop("sl_no",axis=1)
df=df.drop("salary",axis=1)
df.head()
df["gender"]=df["gender"].astype("category")
df["ssc_b"]=df["ssc_b"].astype("category")
df["hsc_b"]=df["hsc_b"].astype("category")
df["degree_t"]=df["degree_t"].astype("category")
df["workex"]=df["workex"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["status"]=df["status"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df.dtypes
df["gender"]=df["gender"].cat.codes
df["ssc_b"]=df["ssc_b"].cat.codes
df["hsc_b"]=df["hsc_b"].cat.codes
df["degree_t"]=df["degree_t"].cat.codes
df["workex"]=df["workex"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
df["status"]=df["status"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
## display dependent variables
y
theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,Y):
    h=sigmoid(x.dot(theta))
    return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-Y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)
print(y_pred)
print(y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

### Placement Dataset
![Screenshot 2025-04-04 101316](https://github.com/user-attachments/assets/6f8899cb-83b4-4000-905b-2e0a39ea4cae)

### Dataset after Feature Engineering
![Screenshot 2025-04-04 101329](https://github.com/user-attachments/assets/7c4d3cd2-cea5-473f-933e-2573a6554fc4)

### Datatypes of Feature column
![Screenshot 2025-04-04 101343](https://github.com/user-attachments/assets/ff599f6b-10c9-4b51-9442-4bae961aa998)

### Dataset after Encoding
![Screenshot 2025-04-04 101402](https://github.com/user-attachments/assets/db1ecf83-417b-4bbf-a18a-36924ac361fd)

### Y Values
![Screenshot 2025-04-04 101417](https://github.com/user-attachments/assets/1a85c693-9785-4def-abc3-2b4602cf70b6)

### Accuracy
![Screenshot 2025-04-04 101441](https://github.com/user-attachments/assets/fd98934f-a112-4bbc-947e-8722434ae64b)

### Y Predicted
![Screenshot 2025-04-04 101518](https://github.com/user-attachments/assets/6fae9842-24b0-48db-b867-08282b36e1df)

### Y Values
![Screenshot 2025-04-04 101533](https://github.com/user-attachments/assets/599bad22-65ce-4c6d-a67e-9f1a77f3565a)

### Y Predicted with different X Values
![Screenshot 2025-04-04 101547](https://github.com/user-attachments/assets/9bb91fe0-8749-4025-b83e-962f6c8fc4f6)
![Screenshot 2025-04-04 101603](https://github.com/user-attachments/assets/dab5e903-6eab-4198-a6d8-cc8ebc52f3b4)


## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


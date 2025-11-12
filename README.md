# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and print head,info of the dataset

2. check for null values

3. Import kmeans and fit it to the dataset

4. Plot the graph using elbow method

5. Print the predicted array

6. Plot the customer segments
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: AHAMADH SULAIMAN M
RegisterNumber: 212224230009
*/

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data= pd.read_csv("Mall_Customers.csv")


df.head()
df.info()
df.isnull().sum()

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])
y_pred = km.predict(data.iloc[:, 3:])
y_pred

data["cluster"] = y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]

plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:

<img width="865" height="285" alt="image" src="https://github.com/user-attachments/assets/0228422d-41b0-4c4c-a428-1ddcc2948b1b" />

<img width="551" height="294" alt="image" src="https://github.com/user-attachments/assets/0d514374-78f8-42c1-a17c-2aec7302c753" />

<img width="397" height="179" alt="image" src="https://github.com/user-attachments/assets/5967dfe4-b27d-4b5a-8712-6ee7d9720acb" />

<img width="1048" height="643" alt="image" src="https://github.com/user-attachments/assets/2dd579e5-d77b-4481-92d6-9ea3edd3c6ac" />

<img width="774" height="260" alt="image" src="https://github.com/user-attachments/assets/c8419189-86e1-461c-9488-c3a60a7c69cb" />

<img width="976" height="624" alt="image" src="https://github.com/user-attachments/assets/d8583f68-888b-449f-9c5d-fbee8f4835b8" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

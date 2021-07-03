# -*- coding: utf-8 -*-
"""K_means.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BibPXEO9pbzJIm3jM0r7dtioJ9LpZIVi

**Author:** Prof. Priyanka Shahane.

**Problem Statement:** To classify the flower species based on petal length & petal width using K-means algorithm of unsupervised machine learning.

**Technical Stack**: Scikit Learn, Pandas, Matplotlib
"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
# %matplotlib inline

"""**Step 1 -** Loading the dataset"""

df = pd.read_csv("Iris.csv")
df.head(10)

"""**Step 2 -** Visualizing the input data """

plt.scatter(df.PetalWidthCm,df.PetalLengthCm)
plt.xlabel('Petal Width')
plt.ylabel('Petal Length')

"""**Step 3** - Identifying optimum number of clusters using Elbow plot method"""

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['PetalWidthCm','PetalLengthCm']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

"""**Step 4** - Applying K means algorithm"""

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['PetalWidthCm','PetalLengthCm']])
y_predicted

df['cluster']=y_predicted
df.head()

km.cluster_centers_

"""**Step 5 -** Visualising clusters"""

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.PetalWidthCm,df1.PetalLengthCm,color='green')
plt.scatter(df2.PetalWidthCm,df2.PetalLengthCm,color='red')
plt.scatter(df3.PetalWidthCm,df3.PetalLengthCm,color='Teal')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()

"""**Step 6** - Pre-processing using min-max scalar"""

scaler = MinMaxScaler()
scaler.fit(df[['PetalLengthCm']])
df['PetalLengthCm'] = scaler.transform(df[['PetalLengthCm']])

scaler.fit(df[['PetalWidthCm']])
df['PetalWidthCm'] = scaler.transform(df[['PetalWidthCm']])

df.head()

plt.scatter(df.PetalLengthCm,df.PetalWidthCm)

"""
**Step 7** - Applying K means algorithm


"""

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['PetalLengthCm','PetalWidthCm']])
y_predicted

df['cluster']=y_predicted
df.head()

km.cluster_centers_

"""**Step 8** - Visualising clusters"""

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.PetalWidthCm,df1.PetalLengthCm,color='green')
plt.scatter(df2.PetalWidthCm,df2.PetalLengthCm,color='red')
plt.scatter(df3.PetalWidthCm,df3.PetalLengthCm,color='Teal')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()


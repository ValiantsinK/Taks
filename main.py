import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

cust_df = pd.read_csv("for_clustering.csv") #считываю файл
df = cust_df.drop(columns=['Age_group', 'Gender']) #дропаю стринговые колонки
data_array = df.values[:, 1:] #перевожу данные в массив и записываю в переменную
data_array = np.nan_to_num(data_array) #если есть поля nan, заменяю их на 0
data_array = StandardScaler().fit_transform(data_array) #стандартизация
clusterNum = 3 #задается количество кластеров
k_means = KMeans(n_clusters=clusterNum, n_init=12) #присвоение переменной k_means класса KMeans
k_means.fit(data_array) #кластеризация
labels = k_means.labels_ #записываем в переменную кластеры
df["Cluster"] = labels #создаём новый столбец Cluster и записываем туда лейблы кластера
df.to_csv('clustered_dataset.csv', header=True) #экспорт в CSV кластеризованного датасета
df.groupby('Cluster').mean().to_csv('grouped_by_clusters.csv',header=True) #импорт в CSV средних значений по кластерам
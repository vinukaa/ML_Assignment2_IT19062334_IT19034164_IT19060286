import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

df=pd.read_csv('D:/SLIIT/4 Year First Semester/ML_Assignment2_IT19034164_IT19062334_IT19060286/car data.csv')

df.head()

df.isnull().sum()

label_encoder = preprocessing.LabelEncoder()

fig=plt.figure(figsize=(8,4))
sns.distplot(df['Selling_Price'])
plt.title('Sales data distribution')


sns.set_theme(style="white")
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 9)),columns=list(df))
corr = d.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})







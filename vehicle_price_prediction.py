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

print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())

final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


final_dataset.head()

final_dataset.drop(['Year'],axis=1,inplace=True)

final_dataset.head()

final_dataset.drop(['Current_Year'],axis=1,inplace=True)

final_dataset.head()

  
  
# Encode labels in column 'species'.
final_dataset['Fuel_Type']= label_encoder.fit_transform(final_dataset['Fuel_Type'])
final_dataset['Seller_Type']= label_encoder.fit_transform(final_dataset['Seller_Type'])
final_dataset['Transmission']= label_encoder.fit_transform(final_dataset['Transmission'])

final_dataset['Fuel_Type']= label_encoder.fit_transform(final_dataset['Fuel_Type'])
  
final_dataset['Fuel_Type'].unique()

final_dataset.corr()

corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

y=final_dataset['Selling_Price']
y.head()
X=final_dataset.drop(['Selling_Price'],axis=1)
X.head()

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

random_forest_regressor=RandomForestRegressor()
random_forest_regressor.fit(X_train,y_train)

train_acc=random_forest_regressor.score(X_train,y_train)
test_acc=random_forest_regressor.score(X_test,y_test)
print('Training Accuracy: ',round(train_acc*100, 2),'%')
print('Testing Accuracy: ',round(test_acc*100, 2),'%')





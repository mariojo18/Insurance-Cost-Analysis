import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'

#*Data ingestion
df = pd.read_csv(path,header=None)
headers = ["age","gender","bmi","no_of_children","smoker","region","charges"]
df.columns = headers
df.replace('?',np.nan,inplace=True)
missing_data = df.isnull()
print(missing_data.head(5))
for column in missing_data.columns.tolist():
        print(column)
        print(missing_data[column].value_counts())
        print("")
#*Data wranling
print(df.info())

df['gender_label']= df['gender'].map({1:'Male',2:'Female'})
df['region_label']= df['region'].map({1:'NW',2:'NE',3:'SW',4:'SE'})

print(df['age'].unique())
df['age']=pd.to_numeric(df['age'],errors='coerce')
df['age']=  df['age'].fillna(df['age'].mean())
print(df['age'].unique())


print(df['smoker'].unique())
df['smoker'].fillna(df['smoker'].mode()[0],inplace= True)
df['smoker']= df['smoker'].astype(int)
df['smoker_label']= df['smoker'].map({0:'smoker',1:'nosmoker'})
print(df['smoker'].unique())
print(df.info())

print(df['charges'].head(5))
df['charges']=df['charges'].round(2)
print(df['charges'].head(5))

#*EDA
sns.regplot(x="bmi", y="charges", data=df, line_kws={"color":"blue"})
plt.ylim(0,)
plt.show()
sns.boxplot(x="smoker",y="charges",data=df)
corr= df.select_dtypes(include=['int64','float']).corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()

#*Model Development

x=df[['charges']]
y=df['smoker']
lrcs=LogisticRegression()
lrcs.fit(x,y)
print(lrcs.score(x,y)) 
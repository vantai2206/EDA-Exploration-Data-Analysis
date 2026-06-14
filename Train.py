
import pandas as pd
import numpy as np
from pandas.core.groupby import categorical

    # Categorical : Survived,Sex,Embarked,Pclass,SibSp,Parch
    # Numerical : (continuous)Age , Fare ( discrete)
    # Mix Type of data: Age, Ticket , Cabin
    # Contain Errors    : Name
    # Blank or Null : Cabin > Ticket > Age > Embarked
# lay du liệu train
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# biet ten tung columns
print(train.columns)
print(test.columns)
# hien thi nhung the firts row
print(train.head())
print(test.head())
# Gop PassengerID thanh index
train.set_index(train.PassengerId,inplace=True)
train.drop("PassengerId",axis=1,inplace=True)
#kieu du lieu trong table
print(train.info())
#thay doi kieu dư lieu
train['Survived'] = train['Survived'].astype('category')
#Thay doi kieu du lieu cung luc
convert = ['Sex','Pclass','SibSp','Parch']
def con_vert(df,convert):
    for col in convert:
        df[col] = df[col].astype('category')
con_vert(train,convert)
print(train.info())
#mo ta cai numerical features value
print(train.describe())
#mo ta cai categorical feature value
print(train.describe(include=['category']))
# Kiem tra gia tri null in the table
print(train.isnull().sum())


import pandas as pd
import numpy as np
from pandas.core.groupby import categorical
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.reshape import tile

# Categorical : Survived,Sex,Embarked,Pclass,SibSp,Parch
    # Numerical : (continuous)Age , Fare ( discrete)
    # Mix Type of data: Age, Ticket , Cabin
    # Contain Errors    : Name
    # Blank or Null : Cabin > Ticket > Age > Embarked

# lay du liệu train
train = pd.read_csv("train.csv")

# biet ten tung columns
print(train.columns)

# hien thi nhung the firts row
print(train.head())

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

#Xem có bao nhiêu người sống sót trên tàu
print(train['Survived'].value_counts().to_frame())

#Bao nhiêu phần trăm số người sống sót
print(train['Survived'].value_counts(normalize=True).to_frame())

#Seaborn
#sns.countplot(data=train,x="Survived",hue='Sex')
#plt.show()

#Xem tất cả các cột categorical sống sót như nào>
col = ['Pclass','SibSp','Sex','Embarked','Parch']
n_row = 2
n_col = 3
fig , ax = plt.subplots(n_row,n_col,figsize=(n_col*3.5,n_row*3.5))
for r in range(0,n_row):
    for c in range(0,n_col):
        i = r*n_col+c
        if i < len(col):
            ax_i = ax[r,c]
            sns.countplot(data= train , x =col[i],hue = 'Survived',ax = ax_i)
            ax_i.set_title(f"Figure {i+1}: Survived {col[i]}")
            ax_i.legend(title='',loc='upper right',labels=['Not Survived','Survived'])
ax.flat[-1].set_visible(False)
plt.tight_layout()
plt.show()
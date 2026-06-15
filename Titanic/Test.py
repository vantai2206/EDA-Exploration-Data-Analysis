import pandas as pd
import numpy as np
from pandas.core.groupby import categorical
# category : Pclass ,Sex,SibSp,Parch
# Numerical : Age,Ticket , Cabin
# Mix type of data : Age , Ticket , Cabin
test = pd.read_csv("test.csv",index_col="PassengerId")
#Chuyển đổi kiểu dữ liệu
Convert = ['Pclass','Sex','SibSp','Parch']
def con_vet(df,convert):
    for col in convert:
        df[col] = df[col].astype('category')
con_vet(test,Convert)
# xem kiểu dữ liệu
print(test.info())


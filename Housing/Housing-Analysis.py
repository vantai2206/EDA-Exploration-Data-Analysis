import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Phân tích SalePrice (target)
train = pd.read_csv('../Data Train/Housing/train_cleaned.csv')
#xem Min vs Max
print(train['SalePrice'].describe())
print("\n")
#Skewness : xem độ lệch của dataset
print(train['SalePrice'].skew())
print("\n")
#Mean vs Median
print(train['SalePrice'].median())
print(train['SalePrice'].mean())
print("\n")
#log-transform xem skewness cải thiện ra sao, nó sẽ nén lại ve sát giá trị 0buo
log_price = np.log1p(train['SalePrice'])
print(log_price.skew())
#Vẽ histogram
sns.histplot(train['SalePrice'],kde=True)
plt.title('Phân phối SalePrice')
plt.show()

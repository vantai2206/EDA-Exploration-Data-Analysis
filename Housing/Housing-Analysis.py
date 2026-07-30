import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#--BƯỚC 1: Phân tích SalePrice---
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
#plt.show()
# Vẽ thêm histogram của log1p để so sánh
sns.histplot(log_price,kde=True)
plt.title("Phân phối SalePrice sau log-transform")
#plt.show()

#--BƯỚC 2 : Tìm insight ẩn---
#Correlation : tìm biến quan trọng để train model
corr = train.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
print(corr)

# Phát hiện outlier
plt.figure(figsize=(8, 6))
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])
plt.title("GrLivArea vs SalePrice")
plt.xlabel("Diện tích Sàn")
plt.ylabel("SalePrice")
plt.show()
# Đã phát hiện ra outlier thì lọc ra
print("\n")
outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)]
print(outliers[['GrLivArea', 'SalePrice']])

#--BƯỚC 3: Visualize - Bắt đầu với heatmap correlation
top_corr_cols = corr.head(10).index
plt.figure(figsize=(10,8))
sns.heatmap(train[top_corr_cols].corr(), annot=True,cmap="Blues",fmt=".2f")
plt.title('Heatmap correlation - Top 10 biến liên quan SalePrice')
plt.show()

#--Bước 4 : Boxplot cho biến categorical
plt.figure(figsize=(14,6))
sns.boxplot(x="Neighborhood", y="SalePrice", data=train)
plt.xticks(rotation=90)
plt.title("Phân phối SalePrice theo Neighborhood")
plt.tight_layout()
plt.show()
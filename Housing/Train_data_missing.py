import pandas as pd

# Bước 1: Đọc dữ liệu
train = pd.read_csv('../Data Train/Housing/train.csv')

# Bước 2: Phân loại biến categorical / numerical
cat_cols = list(train.select_dtypes(include='object').columns)
cat_cols.append('MSSubClass')  # số nhưng là mã phân loại, không phải đại lượng


def convert_category(df, cols):
    for col in cols:
        df[col] = df[col].astype('category')


convert_category(train, cat_cols)
print(train.info())

# Bước 3: Xử lý missing values

# Xem tổng quan các cột đang thiếu dữ liệu
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

#  cột mà NaN nghĩa là "không có đặc điểm này" -> điền 'None'
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType']

for col in none_cols:
    train[col] = train[col].cat.add_categories('None').fillna('None')

#Kiểm tra lại sau khi xử lý nhóm 'None'
print("\n")
missing_after = train.isnull().sum()
print(missing_after[missing_after > 0].sort_values(ascending=False))
# Xử lý các cột thiếu dữ liệu thực sự
train['LotFrontage']= train['LotFrontage'].fillna(train['LotFrontage'].median())
train['GarageYrBlt']= train['GarageYrBlt'].fillna(0)
train['MasVnrArea']= train['MasVnrArea'].fillna(0)
train['Electrical']= train['Electrical'].fillna(train['Electrical'].mode()[0])
#kiêm tra các cột thiếu dữ liệu thật
print('\n')
missing_final = train.isnull().sum()
print(missing_final[missing_final > 0].sort_values(ascending=False))
# lưu dữ liệu
train.to_csv('../Data Train/Housing/train_cleaned.csv', index=False)
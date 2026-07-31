import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
Ý nghĩa của dataset : có 150 bông hoa, 
chia đều 3 loại Iris setosa, Iris versicolor, Iris virginica (mỗi loài 50 mẫu).
- Mỗi mẫu đo 4 đặc trưng (numerical, đơn vị cm):
  - sepal length (chiều dài đài hoa)
  - sepal width (chiều rộng đài hoa)
  - petal length (chiều dài cánh hoa)
  - petal width (chiều rộng cánh hoa)
  Cột thứ 5 là species — nhãn loài (categorical, chính là target nếu làm bài toán phân loại)
"""

# Bước 1 : Đọc dữ liệu
train = pd.read_csv('../Data Train/Iris/Iris.csv')
print(train.head())
print(train.info())
"""Data sạch , không cần phân loại , không thấy NULL"""
#Bước 2: kiểm tra cân bằng lớp
print('\n')
print(train['Species'].value_counts())
"""Chúng ta có 150 bông hoa và có 3 loại hoa chính. Mỗi loại hoa đều có tổng là 50"""
mean=train.groupby('Species')
print(mean.mean())
print("\n")
print(mean.describe())
"""Theo dữ liệu thì:
- Iris-setosa : chiều cao: 1.4 và chiều rộng : 0.2 của cánh hoa , cánh hoa khá là nhỏ
- Iris-versicolor: chiều cao : 4.2 và chiều rộng :1.3 của cánh hoa, cánh hoa có mức độ trung bình
- Iris-Virginica : chiều cao : 5.5 và chiều rộng :2.2 của cánh hoa, cánh hoa có mức độ cao nhất
=> Đây là cách nhận dạng loại hoa"""
#Bước 3 : Phân tích correlation
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.show()
"""
-- PetalLengthCm & PetalWidthCm = 0.962757 — tương quan cực mạnh (gần 1). 
 Nghĩa là: cánh hoa càng dài thì gần như chắc chắn càng rộng theo, gần như tỷ lệ thuận hoàn toàn.
 Đây là cặp tương quan mạnh nhất trong dataset.
-- SepalLengthCm & PetalLengthCm = 0.871754 — tương quan mạnh. 
 Hoa có đài dài thì cánh cũng thường dài theo (dù không chặt bằng cặp trên).
-- SepalLengthCm & PetalWidthCm = 0.817954 — cũng tương quan mạnh, cùng logic: 
 kích thước tổng thể của hoa (đài + cánh) có xu hướng "lớn cùng nhau".
 --SepalWidthCm với các cột còn lại (-0.397729, -0.109369, -0.420516, -0.356544) — 
  đây là điểm khác biệt thú vị: toàn bộ đều âm (dù yếu, dao động -0.1 đến -0.42). 
  Nghĩa là chiều rộng đài hoa có xu hướng ngược chiều với các kích thước còn lại — 
  hoa nào có đài rộng thì thường lại có cánh/đài dài ngắn hơn một chút (và ngược lại). 
  Đây là cột duy nhất "lệch pha" so với phần còn lại của dataset
"""
# Bước 4 : Vẽ Boxplot đề tìm kiếm outlier
# sns.boxplot(x='Species',y='SepalLengthCm',data=train)
# plt.show()
# sns.boxplot(x='Species',y='SepalWidthCm',data=train)
# plt.show()
sns.boxplot(x='Species',y='PetalLengthCm',data=train)
plt.show()
# sns.boxplot(x='Species',y='PetalWidthCm',data=train)
# plt.show()
"""
Nhận xét theo từng loài:
1. SepalLengthCm: không có outlier. Nhưng đáng chú ý: hộp của versicolor (~5.6-6.3) và virginica (~6.2-6.9) chồng lấn nhau khá nhiều 
 khác hẳn với petal, khó dùng riêng cột này để phân biệt 2 loài đó.
2. SepalWidthCm: virginica có 2 outlier (1 điểm cao ~3.8, 1 điểm thấp ~2.0). Đây là cột chồng lấn nhiều nhất trong cả 4 cột 
 cả 3 hộp gần như đè lên nhau (2.5-3.2 cả 3 loài đều có mặt).
 Đúng như dự đoán từ bước correlation: SepalWidthCm là đặc trưng yếu nhất để phân biệt loài.
3. PetalLengthCm: (đã phân tích) — 3 outlier ở setosa, 1 ở versicolor, nhưng 3 hộp tách biệt hoàn toàn, không chồng lấn.
4. PetalWidthCm: setosa có 2 outlier (~0.5-0.6). 3 hộp cũng tách biệt rõ ràng, gần như không chồng lấn — giống petal length..
"""
# Bước 5 : histogram xem phân phối từng cột
sns.histplot(data=train, x='SepalLengthCm', hue='Species', kde=True)
plt.show()
sns.histplot(data=train, x='SepalWidthCm', hue='Species', kde=True)
plt.show()
sns.histplot(data=train, x='PetalWidthCm', hue='Species', kde=True)
plt.show()
sns.histplot(data=train, x='PetalLengthCm', hue='Species', kde=True)
plt.show()
"""
SepalLengthCm: 3 đường cong chồng lấn khá nhiều, nhất là versicolor (cam, đỉnh ~5.8) và virginica (xanh lá, đỉnh ~6.5) 
— 2 đường này giao nhau trên một đoạn rộng (5.5-7). Setosa (xanh dương, đỉnh ~5.0) tách biệt hơn nhưng vẫn có chồng lấn nhẹ với phần đầu của versicolor quanh 5.0-5.5.

SepalWidthCm: đúng như bạn dự đoán — chồng lấn nhiều nhất. 
Versicolor và virginica gần như trùng khít lên nhau hoàn toàn (cả 2 đỉnh quanh 2.8-3.0, hình dạng gần giống hệt nhau)
 — nếu chỉ dựa vào cột này, gần như không thể phân biệt được versicolor với virginica.
  Setosa có nhích cao hơn một chút (đỉnh ~3.3-3.5) nhưng vẫn chồng lấn kha khá với 2 loài kia.

PetalWidthCm: giống hệt pattern của PetalLengthCm 
— setosa tách biệt hoàn toàn (0.1-0.4), versicolor (1.0-1.8) và virginica (1.4-2.5) chỉ chồng lấn nhẹ ở đoạn 1.4-1.8."""
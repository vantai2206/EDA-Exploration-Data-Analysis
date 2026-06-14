import pyodbc
import pandas as pd  # <-- Đã sửa: thêm 's' vào pandas
from sqlalchemy import create_engine

try:
    # 1. Khai báo chuỗi kết nối chuẩn
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-K2K900V\\SQLEXPRESS;"
        "DATABASE=EDA;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

    # 2. Sử dụng SQLAlchemy Engine để Pandas đọc dữ liệu mượt mà, không bị lỗi
    connection_url = f"mssql+pyodbc:///?odbc_connect={connection_string}"
    engine = create_engine(connection_url)

    # 3. Viết query và đọc dữ liệu bằng Pandas
    query = "SELECT * FROM test"
    train = pd.read_sql(query, engine)  # Pass engine vào thay vì conn
    # 4. Hiển thị kết quả
    print("🎉 Kết nối thành công! Dưới đây là 5 dòng đầu tiên:")
    print(train.columns)
    print(train.info())

except Exception as e:
    print("❌ Có lỗi xảy ra rồi:")
    print(e)
import sqlite3
import os

# Kết nối tới SQLite
conn = sqlite3.connect("attendance_system.db")
cursor = conn.cursor()

# Tạo bảng
cursor.execute('''CREATE TABLE IF NOT EXISTS sinhvien (
    id_sinhvien TEXT PRIMARY KEY,
    ten TEXT NOT NULL,
    lop_sinh_hoat TEXT,
    email TEXT,
    anh_thu_muc TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS lop (
    id_lop TEXT PRIMARY KEY,
    ten_lop TEXT NOT NULL,
    tong_sinh_vien INTEGER NOT NULL,
    hoc_ky TEXT,
    nam_hoc TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS diemdanh (
    id_diemdanh INTEGER PRIMARY KEY AUTOINCREMENT,
    id_sinhvien TEXT NOT NULL,
    id_lop TEXT NOT NULL,
    ngay DATE NOT NULL,
    trang_thai TEXT NOT NULL,
    FOREIGN KEY (id_sinhvien) REFERENCES sinhvien (id_sinhvien),
    FOREIGN KEY (id_lop) REFERENCES lop (id_lop)
)''')

# Tạo hoặc cập nhật VIEW
cursor.execute('''
CREATE VIEW IF NOT EXISTS sinhvien_theo_lop AS
SELECT 
    lop.id_lop, 
    lop.ten_lop, 
    sinhvien.id_sinhvien, 
    sinhvien.ten, 
    sinhvien.email,
    COUNT(CASE WHEN diemdanh.trang_thai = 'Vắng' THEN 1 END) AS so_lan_vang
FROM sinhvien
JOIN diemdanh ON sinhvien.id_sinhvien = diemdanh.id_sinhvien
JOIN lop ON diemdanh.id_lop = lop.id_lop
GROUP BY lop.id_lop, sinhvien.id_sinhvien;
''')

conn.commit()

# Thêm sinh viên và tự động tạo thư mục ảnh
def them_sinhvien(id_sinhvien, ten, lop_sinh_hoat, email):
    thu_muc_anh = f"./image/sinhvien/{id_sinhvien}"
    os.makedirs(thu_muc_anh, exist_ok=True)  # Tạo thư mục lưu ảnh
    
    cursor.execute("INSERT INTO sinhvien (id_sinhvien, ten, lop_sinh_hoat, email, anh_thu_muc) VALUES (?, ?, ?, ?, ?)",
                   (id_sinhvien, ten, lop_sinh_hoat, email, thu_muc_anh))
    conn.commit()
    print(f"Đã thêm sinh viên {ten} và tạo thư mục: {thu_muc_anh}")


def them_lop(id_lop, ten_lop, tong_sinh_vien, hoc_ki,nam_hoc):
   
    cursor.execute("INSERT INTO lop (id_lop, ten_lop, tong_sinh_vien, hoc_ki, nam_hoc) VALUES (?, ?, ?, ?, ?)",
                   (id_lop, ten_lop, tong_sinh_vien, hoc_ki, nam_hoc))
    conn.commit()
    print(f"Đã đăng kí lớp mới là : {ten_lop}, học kì: {hoc_ki}, nam_hoc{nam_hoc}")

#################################################################
# Hàm lấy danh sách sinh viên theo lớp và số lần vắng mặt
def lay_danh_sach_sinh_vien_theo_lop(id_lop):
    # Kết nối đến cơ sở dữ liệu
    conn = sqlite3.connect("attendance_system.db")
    cursor = conn.cursor()

    # Truy vấn từ VIEW
    cursor.execute('''
    SELECT id_lop, ten_lop, id_sinhvien, ten, email, so_lan_vang
    FROM sinhvien_theo_lop
    WHERE id_lop = ?;
    ''', (id_lop,))

    # Lấy kết quả
    danh_sach = cursor.fetchall()

    # Đóng kết nối
    conn.close()

    return danh_sach
'''
# Sử dụng hàm
id_lop_can_tim = "lop_01"
danh_sach_sinh_vien = lay_danh_sach_sinh_vien_theo_lop(id_lop_can_tim)

# In kết quả
print("Danh sách sinh viên của lớp:", id_lop_can_tim)
for sv in danh_sach_sinh_vien:
    print(f"ID: {sv[2]}, Tên: {sv[3]}, Email: {sv[4]}, Số lần vắng: {sv[5]}")
'''
import os
from flask import Flask, render_template, Response, request, jsonify
import sqlite3
import cv2

app = Flask(__name__)

video_capture = None
camera_status = {"is_active": False}  # Trạng thái camera

def start_camera():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

def stop_camera():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None

def generate_frames():
    global video_capture
    while camera_status["is_active"]:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
###############################################################################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global video_capture
    if not camera_status["is_active"]:
        return "", 204  # Không trả về feed nếu camera không được bật
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_camera', methods=['POST'])
def control_camera():
    global camera_status
    action = request.json.get("action", "")
    if action == "start":
        if not camera_status["is_active"]:
            start_camera()
            camera_status["is_active"] = True
    elif action == "stop":
        if camera_status["is_active"]:
            stop_camera()
            camera_status["is_active"] = False
    return {"status": "success"}




####################################################################################################
DB_PATH = "attendance_system.db"

def execute_query(query, params=()):
    """Thực thi câu truy vấn và trả kết quả."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.fetchall()


####


@app.route("/api/register_class", methods=["POST"])
def register_class():
    data = request.json
    try:
        execute_query(
            '''INSERT INTO lop (id_lop, ten_lop, tong_sinh_vien, hoc_ky, nam_hoc)
               VALUES (?, ?, ?, ?, ?)''',
            (data["id_lop"], data["ten_lop"], data["tong_sinh_vien"], data["hoc_ky"], data["nam_hoc"]),
        )
        return jsonify({"message": "Class registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Class ID already exists"}), 400

########

@app.route("/api/register_student", methods=["POST"])
def register_student():
    data = request.json
    id_sinhvien = data["id_sinhvien"]
    # Đường dẫn lưu ảnh
    anh_thu_muc = f"./image/sinhvien/{id_sinhvien}/{id_sinhvien}.jpg"
    
    try:
        # Tạo thư mục ảnh nếu chưa tồn tại
        os.makedirs(os.path.dirname(anh_thu_muc), exist_ok=True)

        # Lưu thông tin sinh viên vào database
        execute_query(
            '''INSERT INTO sinhvien (id_sinhvien, ten, lop_sinh_hoat, email, anh_thu_muc)
               VALUES (?, ?, ?, ?, ?)''',
            (id_sinhvien, data["ten"], data["lop_sinh_hoat"], data["email"], anh_thu_muc),
        )
        return jsonify({"message": "Student registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Student ID already exists"}), 400

"""
@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    data = request.json
    id_sinhvien = data.get("id_sinhvien")  # Lấy id_sinhvien từ request

    if not id_sinhvien:
        return jsonify({"success": False, "error": "Missing student ID"}), 400

    # Đường dẫn lưu ảnh
    thu_muc_anh = f"./image/sinhvien/{id_sinhvien}"
    filename = os.path.join(thu_muc_anh, f"{id_sinhvien}.jpg")
    
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(thu_muc_anh, exist_ok=True)

        # Mở camera để chụp ảnh
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(filename, frame)  # Lưu ảnh vào đường dẫn
            return jsonify({"success": True, "path": filename})
        else:
            return jsonify({"success": False, "error": "Cannot capture photo"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
"""

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    data = request.json
    id_sinhvien = data.get("id_sinhvien")  # Lấy id_sinhvien từ request

    if not id_sinhvien:
        return jsonify({"success": False, "error": "Missing student ID"}), 400

    # Đường dẫn lưu ảnh
    thu_muc_anh = f"./image/sinhvien/{id_sinhvien}"
    
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(thu_muc_anh, exist_ok=True)

        # Mở camera để chụp ảnh
        import cv2
        cap = cv2.VideoCapture(0)

        paths = []  # Danh sách lưu đường dẫn các ảnh đã chụp
        for i in range(1, 31):  # Chụp 6 ảnh
            ret, frame = cap.read()
            if ret:
                filename = os.path.join(thu_muc_anh, f"{id_sinhvien}_{i}.jpg")
                cv2.imwrite(filename, frame)  # Lưu ảnh vào đường dẫn
                paths.append(filename)  # Thêm đường dẫn vào danh sách
            else:
                cap.release()
                return jsonify({"success": False, "error": "Cannot capture photo"}), 500

        cap.release()
        return jsonify({"success": True, "paths": paths})  # Trả về danh sách đường dẫn ảnh đã chụp
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


#########

@app.route("/api/classes", methods=["GET"])
def get_classes():
    classes = execute_query("SELECT * FROM lop")
    return jsonify([{"id_lop": c[0], "ten_lop": c[1], "tong_sinh_vien": c[2], "hoc_ky": c[3], "nam_hoc": c[4]} for c in classes])

##########
@app.route("/api/students_in_class/<id_lop>", methods=["GET"])
def get_students_in_class(id_lop):
    students = execute_query(
        '''SELECT lop.id_lop, lop.ten_lop, sinhvien.id_sinhvien, sinhvien.ten, sinhvien.email,
                  COUNT(CASE WHEN diemdanh.trang_thai = 'Vắng' THEN 1 END) AS so_lan_vang
           FROM sinhvien
           JOIN diemdanh ON sinhvien.id_sinhvien = diemdanh.id_sinhvien
           JOIN lop ON diemdanh.id_lop = lop.id_lop
           WHERE lop.id_lop = ?
           GROUP BY sinhvien.id_sinhvien''',
        (id_lop,),
    )
    return jsonify([{"id_lop": s[0], "ten_lop": s[1], "id_sinhvien": s[2], "ten": s[3], "email": s[4], "so_lan_vang": s[5]} for s in students])


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "Missing search query"}), 400

    # Tìm kiếm theo id_lop hoặc id_sinhvien
    students = execute_query(
        '''SELECT lop.id_lop, lop.ten_lop, lop.tong_sinh_vien, lop.hoc_ky, lop.nam_hoc, sinhvien.id_sinhvien, sinhvien.ten, sinhvien.email,
                  COUNT(CASE WHEN diemdanh.trang_thai = 'Vắng' THEN 1 END) AS so_lan_vang
           FROM lop
           LEFT JOIN diemdanh ON lop.id_lop = diemdanh.id_lop
           LEFT JOIN sinhvien ON sinhvien.id_sinhvien = diemdanh.id_sinhvien
           WHERE lop.id_lop LIKE ? OR sinhvien.id_sinhvien LIKE ?
           GROUP BY lop.id_lop, sinhvien.id_sinhvien''',
        (f"%{query}%", f"%{query}%"),
    )
    return jsonify([{"id_lop": s[0], "ten_lop": s[1], "tong_sinh_vien": s[2], "hoc_ky": s[3], "nam_hoc": s[4], "id_sinhvien": s[5] or "-", "ten": s[6] or "-", "email": s[7] or "-", "so_lan_vang": s[8] or 0} for s in students])


if __name__ == '__main__':
    app.run(debug=True)

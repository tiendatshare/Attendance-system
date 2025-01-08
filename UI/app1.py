import os
from flask import Flask, render_template, Response, request, jsonify,send_file
import sqlite3
import cv2
import threading
import pandas as pd
import time
import numpy as np
from keras_facenet import FaceNet
import mediapipe as mp
import pickle
import sys
sys.path.append(os.path.dirname(os.path.abspath("Silent_Face_Anti_Spoofing")))
from Silent_Face_Anti_Spoofing.test import test
from sklearn.neighbors import KNeighborsClassifier

import json

app = Flask(__name__)

# Camera setup
video_capture = None
camera_status = {"is_active": False}  # Camera status
camera_active = False
selected_class = None
# Face recognition setup
face_detection = None
facenet = None
knn = None
tracked_faces = {}

# Initialize models
def init_models():
    global face_detection, facenet, knn
    # Initialize MediaPipe for face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Initialize FaceNet and KNN for face recognition
    facenet = FaceNet()
    # Load face embeddings from file
    with open("embeddings.pkl", 'rb') as f:
        embedding_data, embedding_labels = pickle.load(f)
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(embedding_data, embedding_labels)



class VideoCamera:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print("Already started!!")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def release(self):
        self.cap.release()


# is_image_blurry()  loại bỏ các frame bị mờ
def is_image_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100  # Ngưỡng độ mờ (có thể điều chỉnh)


# Functions for database queries
DB_PATH = "attendance_system.db"

def execute_query(query, params=()):
    """Execute a query and return the result."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.fetchall()

def get_student_info(id_sinhvien):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sinhvien WHERE id_sinhvien = ?", (id_sinhvien,))
    student = cursor.fetchone()
    conn.close()
    if student:
        return {
            'id_sinhvien': student[0],
            'ten': student[1],
            'lop_sinh_hoat': student[2],
            'email': student[3]
        }
    return None

def record_attendance(id_sinhvien, id_lop, trang_thai):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    current_date = time.strftime('%Y-%m-%d')
    
    # Check if attendance is already recorded
    cursor.execute("""SELECT * FROM diemdanh WHERE id_sinhvien = ? AND id_lop = ? AND ngay = ?""", (id_sinhvien, id_lop, current_date))
    
    if not cursor.fetchone():
        cursor.execute("""INSERT INTO diemdanh (id_sinhvien, id_lop, ngay, trang_thai) VALUES (?, ?, ?, ?)""", (id_sinhvien, id_lop, current_date, trang_thai))
        conn.commit()
    conn.close()

# IOU function to calculate Intersection over Union
def compute_iou(box1, box2):
    x1, y1, w1, h1 = map(int, box1)
    x2, y2, w2, h2 = map(int, box2)
    
    
    # Tính diện tích phần giao nhau
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

#############    
face_id_counter = 0  # ID tăng dần
def process_frame(frame):
    global face_id_counter, tracked_faces
    if face_detection is None or facenet is None or knn is None:
        return frame, None

    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    attendance_info = None

    ih, iw, _ = frame.shape

    # Update frames since last seen for tracked faces
    for face_id in list(tracked_faces.keys()):
        tracked_faces[face_id]['frames_since_last_seen'] += 1

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            if x < 0 or y < 0 or x + w > iw or y + h > ih or w * h < 500 or w < 50 or h < 50:
                continue

            matched_face_id = None
            max_iou = 0
            for face_id, face_info in tracked_faces.items():
                iou = compute_iou((x, y, w, h), face_info['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    matched_face_id = face_id

            if max_iou > 0.3:
                tracked_faces[matched_face_id]['bbox'] = (x, y, w, h)
                tracked_faces[matched_face_id]['frames_since_last_seen'] = 0
                tracked_faces[matched_face_id]['stable_frames'] += 1
            else:
                tracked_faces[face_id_counter] = {
                    'bbox': (x, y, w, h),
                    'recognized': False,
                    'frames_since_last_seen': 0,
                    'stable_frames': 1,
                    'name': "Unknown"
                }
                face_id_counter += 1

    for face_id, face_info in tracked_faces.items():
        x, y, w, h = face_info['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if face_info['recognized'] else (0, 0, 255), 2)
        student_id = face_info["name"].split('/')[0]
        cv2.putText(frame, f'ID: {face_id} - {student_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if face_info['recognized'] else (0, 0, 255), 2)

        if not face_info['recognized'] and face_info['stable_frames'] % 10 == 0:
            face_roi = frame[max(0, y):min(y + h, ih), max(0, x):min(x + w, iw)]
            if face_roi.size == 0 or is_image_blurry(face_roi):
                continue
            
            cv2.imwrite("temp.jpg", face_roi)
            #---------
            try:
                label, value = test("temp.jpg", "Silent_Face_Anti_Spoofing/resources/anti_spoof_models", 0)
            except TypeError as e:
                print(f"Error occurred: {e}. Continuing execution.")
                label, value = None, None 
            
            if not is_image_blurry(face_roi):
                try:
                    resized_face = cv2.resize(face_roi, (160, 160))
                    normalized_face = np.array(resized_face)
                    input_face = np.expand_dims(normalized_face, axis=0)
                    embedding = facenet.embeddings(input_face)[0]
                    #embedding = facenet.embeddings(np.expand_dims(resized_face, axis=0))[0]

                    distances, indices = knn.kneighbors([embedding])
                    
                    print(f"Distance to closest embedding: {distances[0][0]}")
                    threshold = 0.3
                    if  distances[0][0] < threshold:
                        predicted_name = knn.predict([embedding])[0]
                        student_id = predicted_name.split('/')[0]
                        if label == 1: print(f"{student_id} ís real face ")
                        else: print(f"{student_id} is fake face")
                    else:
                        predicted_name = "Unknown"

                    tracked_faces[face_id]['name'] = student_id
                    tracked_faces[face_id]['recognized'] = True

                    if predicted_name != "Unknown" and label == 1:
                        student_info = get_student_info(student_id)
                        if student_info and selected_class:
                            record_attendance(student_id, selected_class, "Có mặt")
                            attendance_info = {
                                'face_id': f"face_{face_id_counter}",
                                'id_sinhvien': student_id,
                                'ten_sinhvien': student_info['ten'],
                                'is_real': True,
                                'score': float(value)
                            }
                            face_id_counter += 1
                except Exception as e:
                    print(f"Error processing face ID {face_id}: {e}")
            else:
                print(f"Khuôn mặt ID {face_id} bị mờ, bỏ qua.")

    # Remove faces that have not been seen for a while
    for face_id in list(tracked_faces.keys()):
        if tracked_faces[face_id]['frames_since_last_seen'] > 10:
            del tracked_faces[face_id]

    return frame, attendance_info

###########

def generate_frames():
    global video_capture
    while camera_status["is_active"]:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            processed_frame, attendance_info = process_frame(frame)
            
            if attendance_info:
                yield f"data: {json.dumps(attendance_info)}\n\n"
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
#################################################3
# Function to generate frames for the registration page without processing
def generate_simple_frames():
    global video_capture
    while camera_status["is_active"]:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for the registration video feed
@app.route('/registration_video_feed')
def registration_video_feed():
    if not camera_status["is_active"]:
        return "", 205 
    return Response(generate_simple_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#############################################################

# Camera control endpoints
@app.route('/video_feed')
def video_feed():
    if not camera_status["is_active"]:
        return "", 204  
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

# Routes for class and student management
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

@app.route('/api/export_class/<id_lop>', methods=['GET'])
def export_class(id_lop):
    try:
        # Lấy dữ liệu sinh viên theo lớp
        students = execute_query(
            '''SELECT sinhvien.id_sinhvien, sinhvien.ten, sinhvien.email,
                      COUNT(CASE WHEN diemdanh.trang_thai = 'Vắng' THEN 1 END) AS so_lan_vang
               FROM sinhvien
               JOIN diemdanh ON sinhvien.id_sinhvien = diemdanh.id_sinhvien
               WHERE diemdanh.id_lop = ?
               GROUP BY sinhvien.id_sinhvien''',
            (id_lop,)
        )

        # Tạo DataFrame từ dữ liệu
        df = pd.DataFrame(students, columns=["ID Sinh viên", "Tên", "Email", "Số lần vắng"])

        # Tạo file Excel
        file_path = "D:\\Graduation_Project\\UI\\export_class.xlsx"
        df.to_excel(file_path, index=False)

        # Trả về file Excel
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/register_student", methods=["POST"])
def register_student():
    data = request.json
    id_sinhvien = data["id_sinhvien"]
    # Image path
    anh_thu_muc = f"./image/sinhvien/{id_sinhvien}"
    
    try:
        execute_query(
            '''INSERT INTO sinhvien (id_sinhvien, ten, lop_sinh_hoat, email, anh_thu_muc) 
               VALUES (?, ?, ?, ?, ?)''',
            (data["id_sinhvien"], data["ten"], data["lop_sinh_hoat"], data["email"], anh_thu_muc)
        )
        return jsonify({"message": "Student registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Student ID already exists"}), 400

@app.route('/api/select_class', methods=['POST'])
def select_class():
    global selected_class
    data = request.json
    selected_class = data.get('id_lop')
    return jsonify({'success': True, 'message': f'Selected class: {selected_class}'})

##########################################
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

        paths = []  # Danh sách lưu đường dẫn các ảnh đã chụp
        for i in range(1, 31):  # Chụp 30 ảnh
            success, frame = video_capture.read()
            if success:
                filename = os.path.join(thu_muc_anh, f"{id_sinhvien}_{i}.jpg")
                cv2.imwrite(filename, frame)  # Lưu ảnh vào đường dẫn
                paths.append(filename)  # Thêm đường dẫn vào danh sách
            else:
                return jsonify({"success": False, "error": "Cannot capture photo"}), 500

        return jsonify({"success": True, "paths": paths})  # Trả về danh sách đường dẫn ảnh đã chụp
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def reload_model():
    global knn
    with open("embeddings.pkl", 'rb') as f:
        embedding_data, embedding_labels = pickle.load(f)
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(embedding_data, embedding_labels)
@app.route('/run_create_embeddings', methods=['POST'])
def run_create_embeddings():
    try:
        # Chạy script create_embeddings.py
        os.system('python model/create_embeddings.py')
        reload_model()
        return jsonify({"success": True, "message": "Embeddings created successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})



#########
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

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == "__main__":
    init_models()
    
    app.run( debug=True)

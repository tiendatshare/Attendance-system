import os
import numpy as np
import pickle
from keras_facenet import FaceNet
import cv2
import mediapipe as mp

# Khởi tạo mô hình FaceNet
embedder = FaceNet()

# Khởi tạo Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Tải embeddings và nhãn hiện có từ file
def load_existing_embeddings(filename="embeddings.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return np.empty((0, 512)), []  # Trả về mảng rỗng với số chiều phù hợp

# Tạo embedding cho ảnh sinh viên
def create_embeddings_for_students(student_folder, existing_labels):
    embeddings = []
    labels = []

    for student_name in os.listdir(student_folder):
        student_dir = os.path.join(student_folder, student_name)
        
        if os.path.isdir(student_dir):
            for image_name in os.listdir(student_dir):
                image_path = os.path.join(student_dir, image_name)
                
                # Kiểm tra nếu ảnh đã được xử lý
                if f"{student_name}/{image_name}" in existing_labels:
                    continue
                
                image = cv2.imread(image_path)
                
                # Phát hiện khuôn mặt
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = image.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        face = image[y:y+h, x:x+w]
                        
                        # Resize và chuẩn bị ảnh cho FaceNet
                        face = cv2.resize(face, (160, 160))
                        face_array = np.array(face)
                        face_array = np.expand_dims(face_array, axis=0)
                        
                        # Tạo embedding cho ảnh sinh viên
                        embedding = embedder.embeddings(face_array)
                        embeddings.append(embedding[0])
                        labels.append(f"{student_name}/{image_name}")

    return np.array(embeddings), labels

# Lưu embedding và nhãn vào file pickle
def save_embeddings(embeddings, labels, filename="embeddings.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((embeddings, labels), f)

# Tạo embedding cho sinh viên
student_folder = 'D:\\Graduation_Project\\image\\sinhvien'  # Đường dẫn đến thư mục sinh viên
existing_embeddings, existing_labels = load_existing_embeddings()

new_embeddings, new_labels = create_embeddings_for_students(student_folder, existing_labels)

# Kết hợp embeddings và nhãn mới với những cái đã tồn tại
if new_embeddings.size > 0:  # Kiểm tra nếu có embeddings mới
    all_embeddings = np.concatenate((existing_embeddings, new_embeddings), axis=0)
    all_labels = existing_labels + new_labels

    # Lưu lại tất cả embeddings và nhãn
    save_embeddings(all_embeddings, all_labels)
    print("Embeddings updated successfully.")
else:
    print("No new embeddings to update.")

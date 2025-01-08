import pickle

embedding_file = "D:\\Graduation_Project\\embeddings.pkl"

with open(embedding_file, 'rb') as f:
    data = pickle.load(f)
    print(type(data))
    print(data)

###################
import os
import numpy as np
import cv2
import pickle
from keras_facenet import FaceNet
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier

# Khởi tạo mô hình FaceNet và Mediapipe
embedder = FaceNet()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Hàm phát hiện khuôn mặt và cắt ảnh khuôn mặt
def detect_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face = image[y:y+h, x:x+w]
            return face, (x, y, w, h)
    return None, None

# Tải embedding và nhãn từ file pickle
def load_embeddings(filename="embeddings.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Nhận diện khuôn mặt từ ảnh camera
def recognize_face_from_camera(embeddings, labels):
    cap = cv2.VideoCapture(0)
    
    # Khởi tạo mô hình phân loại k-NN
    knn = KNeighborsClassifier(n_neighbors=11, metric='cosine')
    knn.fit(embeddings, labels)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện khuôn mặt từ camera
        face, bbox = detect_face(frame)

        if face is not None:
            # Hiển thị khuôn mặt phát hiện được từ camera
            cv2.imshow('Detected Face', face)  # Hiển thị ảnh khuôn mặt

            face = cv2.resize(face, (160, 160))
            face_array = np.array(face)
            face_array = np.expand_dims(face_array, axis=0)

            # Tạo embedding cho khuôn mặt từ camera
            embedding = embedder.embeddings(face_array)
            
            # Sử dụng k-NN để dự đoán
            predicted_label = knn.predict(embedding)[0]

            # Vẽ bounding box và ghi tên sinh viên
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị ảnh với khuôn mặt đã nhận diện
        cv2.imshow('Face Recognition', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tải embedding đã lưu và thực hiện nhận diện khuôn mặt từ camera
embeddings, labels = load_embeddings()
recognize_face_from_camera(embeddings, labels)


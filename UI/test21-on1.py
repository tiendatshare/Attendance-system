import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("Silent_Face_Anti_Spoofing")))
from Silent_Face_Anti_Spoofing.test import test

import cv2
import mediapipe as mp
import threading
import time
import numpy as np
from collections import deque
import pickle
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier

################################
#face spoofing
spoofs=[]
device=0







################################
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class VideoCaptureAsync:
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

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def is_image_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100  # Threshold for blurriness (adjust as needed)

def load_embeddings(filename="embeddings.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load model FaceNet
facenet = FaceNet()
embedding_data, embedding_labels = load_embeddings()
knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
knn.fit(embedding_data, embedding_labels)

def display_captured_face(face_roi, face_id, name):
    window_name = f"Captured Face ID: {face_id}"
    labeled_face = cv2.putText(face_roi.copy(), name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(window_name, labeled_face)
    cv2.waitKey(1)  # Display for 1 second



cap = VideoCaptureAsync(0).start()
tracked_faces = {}
face_id_counter = 0
max_frames_lost = 10
stable_frame_threshold = 10
prev_frame_time = time.time()

while cap.started:
    ret, image = cap.read()
    if not ret:
        continue

      
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = face_detection.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    ih, iw, _ = image.shape

    for face_id in list(tracked_faces.keys()):
        tracked_faces[face_id]['frames_since_last_seen'] += 1

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)

            # Skip invalid or small bounding boxes
            if x < 0 or y < 0 or x + w > iw or y + h > ih or w * h < 500 or w < 50 or h < 50:
                continue

            matched_face_id = None
            max_iou = 0
            for face_id, face_info in tracked_faces.items():
                iou = calculate_iou((x, y, w, h), face_info['bbox'])
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

    label_spoof = "Unknown"
    for face_id, face_info in tracked_faces.items():
        
        x, y, w, h = face_info['bbox']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0) if face_info['recognized'] else (0, 0, 255), 2)
        student_id = face_info["name"].split('/')[0]
        cv2.putText(image, f'ID: {face_id} - {student_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if face_info['recognized'] else (0, 0, 255), 2)



        if not face_info['recognized'] and face_info['stable_frames'] % 10 == 0:
            face_roi = image[max(0, y):min(y + h, ih), max(0, x):min(x + w, iw)]
            
            if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue  # Skip if ROI is invalid
                          
                
            if not is_image_blurry(face_roi):
                try:
                    resized_face = cv2.resize(face_roi, (160, 160))
                    normalized_face = np.array(resized_face)
                    input_face = np.expand_dims(normalized_face, axis=0)
                    embedding = facenet.embeddings(input_face)[0]

                    distances, indices = knn.kneighbors([embedding])
                    print(f"Distance to closest embedding: {distances[0][0]}")

                    if distances[0][0] > 0.3:  # Ngưỡng khoảng cách, tùy chỉnh theo dữ liệu
                        predicted_name = "Unknown"
                    else:
                        predicted_name = knn.predict([embedding])[0]
                        student_id = predicted_name.split('/')[0]

                    #########################################################
                    #face spoofing
                    # Chuyển đổi face_roi thành định dạng ảnh tạm thời
                    is_success, buffer = cv2.imencode(".jpg", face_roi)
                    if is_success:
                    
                        # Save the resized image
                        cv2.imwrite("temp.jpg", face_roi)
                        #---------
                        label, value =test("temp.jpg","Silent_Face_Anti_Spoofing/resources/anti_spoof_models",device)
                        if label==1:
                            print(f"{student_id} is Real Face. Score: {value:.2f}.")
                        else:
                            print(f"{student_id} is Fake Face. Score: {value:.2f}.")   
                
                #########################################################

                    #tracked_faces[face_id]['name'] = predicted_name
                    #tracked_faces[face_id]['recognized'] = True
                    if predicted_name == "Unknown":
                        tracked_faces[face_id]['name'] = "Unknown"
                        tracked_faces[face_id]['recognized'] = False
                    else:
                        tracked_faces[face_id]['name'] = student_id
                        tracked_faces[face_id]['recognized'] = True


            
                    print(f"Phân biệt khuôn mặt ID: {face_id} - {student_id}")
                    display_captured_face(face_roi, face_id, student_id)
                except Exception as e:
                    print(f"Lỗi xử lý khuôn mặt ID {face_id}: {e}")
            else:
                print(f"Khuôn mặt ID {face_id} bị mờ, bỏ qua.")
            
        

    for face_id in list(tracked_faces.keys()):
        if tracked_faces[face_id]['frames_since_last_seen'] > max_frames_lost:
            del tracked_faces[face_id]

    new_frame_time = time.time()
    if new_frame_time > prev_frame_time:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
    prev_frame_time = new_frame_time

    fps = int(fps)
    fps = str(fps)
    cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Face Detection and Recognition', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cap.release()
cv2.destroyAllWindows()

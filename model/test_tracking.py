import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("sort.py")))
from sort import Sort
import cv2
import mediapipe as mp
from sort import Sort
import numpy as np

# Khởi tạo các đối tượng Mediapipe và SORT
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
tracker = Sort()  # Khởi tạo đối tượng SORT

# Mở camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi ảnh từ BGR sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Khởi tạo một list chứa các box cho các khuôn mặt
    face_boxes = []
    if results.detections:
        for detection in results.detections:
            # Lấy tọa độ bounding box của khuôn mặt
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                          int(bboxC.width * iw), int(bboxC.height * ih)
            face_boxes.append([x, y, x + w, y + h])

    # Kiểm tra nếu face_boxes không rỗng trước khi cập nhật tracker
    if len(face_boxes) > 0:
        face_boxes = np.array(face_boxes)
        trackers = tracker.update(face_boxes)  # Cập nhật tracker với các box khuôn mặt
    else:
        trackers = []

    # Vẽ các bounding box và đánh dấu ID trên ảnh
    for tracker in trackers:
        x1, y1, x2, y2, track_id = tracker
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow('Face Detection and Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
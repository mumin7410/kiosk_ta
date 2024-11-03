# recog.py 
from django.shortcuts import render
import numpy as np
import pandas as pd
import cv2
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from concurrent.futures import ThreadPoolExecutor
import threading
from kiosk.celery import face_recognition
from django.http import StreamingHttpResponse, JsonResponse
from io import StringIO 
import base64

class FaceRecognition:

    def face_detection(self, faces, dataset, image):
        serialized_dataset = dataset.to_dict(orient='records') 
        if len(faces) > 0:
            # Process the first face detected
            for face in faces:
                test_vector = face['embedding'].reshape(1, -1).tolist()[0]
                print("==== triggering face recognition ====")
                _, buffer = cv2.imencode('.jpg', image)
                image_bytes = buffer.tobytes()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                face_recognition.delay(serialized_dataset, test_vector, image_b64)

    def video_capture(self):
        # Connect to Redis
        r = redis.Redis(host='sony-redis', port=6379, password='2vBuMI9QeQ9tMGeG', db=0)
        json_data = r.get('employee_data')
        if json_data is None:
            return JsonResponse({"error": "No data found in Redis"}, status=404)
        json_str = json_data.decode('utf-8')
        df = pd.read_json(StringIO(json_str), dtype={'emp_id': str})
        if df.empty:
            return JsonResponse({"error": "DataFrame is empty"}, status=404)
        # Configure face analysis
        faceapp = FaceAnalysis(providers=['CPUExecutionProvider'])
        faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

        def generate():
            cap = cv2.VideoCapture(0)  # Open webcam
            frame_count = 0
            while True:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) #fix if use WSL then enable this line
                ret, frame = cap.read()  # Capture frame-by-frame
                if not ret:
                    break

                frame_count += 1

                # Process every 5th frame to reduce load
                if frame_count % 5 == 0:
                    # Serialize the frame (convert it to bytes)
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    # Check if frame is successfully decoded
                    if frame is None:
                        print("Error: Frame could not be decoded.")
                        return

                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                    # Detect faces in the frame
                    faces = faceapp.get(small_frame)

                    self.face_detection(faces, df, frame)  # Pass serialized dataset
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            cap.release()

        response = StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')
        return response

if __name__ == "__main__":
    fr = FaceRecognition()
    fr.video_capture()

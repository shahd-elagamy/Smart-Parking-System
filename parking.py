import json
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class ParkingManager:
    def __init__(self, model_path="yolov8n.pt", json_file="bounding_boxes.json"):
        self.model = YOLO(model_path)
        with open(json_file) as f:
            self.parking_spaces = json.load(f)
        self.prev_status = {}  # لتتبع التغيرات السابقة
        self.colors = {
            "available": (0, 255, 0),
            "occupied": (0, 0, 255),
            "centroid": (255, 0, 189),
            "text": (0, 0, 0)
        }

    def log_occupied_spot(self, spot_id, coords, occupied_count, available_count):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("occupied_spots_log.txt", "a") as log_file:
            log_file.write(f"{timestamp} - Spot {spot_id} OCCUPIED - Coords: {coords} - Occupied: {occupied_count} - Available: {available_count}\n")

    def process_frame(self, frame):
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame[0, 0, 0] == frame[0, 0, 2]:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = self.model(frame)
        occupancy = {
            "total": len(self.parking_spaces),
            "occupied": 0,
            "available": len(self.parking_spaces)
        }

        for idx, space in enumerate(self.parking_spaces, start=1):
            space_occupied = False
            pts = np.array(space["points"], dtype=np.int32).reshape((-1, 1, 2))

            for box in results[0].boxes:
                x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

                if cv2.pointPolygonTest(pts, (x_center, y_center), False) >= 0:
                    space_occupied = True
                    cv2.circle(frame, (x_center, y_center), 5, self.colors["centroid"], -1)
                    break

            # إذا حصل تغيير في حالة الموقف، نسجل
            if self.prev_status.get(idx) != space_occupied:
                self.prev_status[idx] = space_occupied
                if space_occupied:
                    self.log_occupied_spot(idx, space["points"], occupancy["occupied"] + 1, occupancy["available"] - 1)

            if space_occupied:
                occupancy["occupied"] += 1
                occupancy["available"] -= 1
                color = self.colors["occupied"]
            else:
                color = self.colors["available"]

            cv2.polylines(frame, [pts], True, color, 2)
            label_position = tuple(pts[0][0])
            cv2.putText(frame, f"{idx}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["text"], 2)

        # معلومات الإشغال
        text1 = f"Occupied spots: {occupancy['occupied']}/{occupancy['total']}"
        text2 = f"Available spots: {occupancy['available']}/{occupancy['total']}"
        text3 = f"Total spots: {occupancy['total']}"
        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors["text"], 2)
        cv2.putText(frame, text2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors["text"], 2)
        cv2.putText(frame, text3, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors["text"], 2)

        return frame, occupancy

# التنفيذ الرئيسي
if __name__ == "__main__":
    manager = ParkingManager(model_path="yolo11s.pt", json_file="bounding_boxes.json")
    cap = cv2.VideoCapture("parking.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, occupancy = manager.process_frame(frame)

        try:
            cv2.imshow("SMART PARKING SYSTEM", processed_frame)
            print(f"Occupied spots: {occupancy['occupied']}/{occupancy['total']}")

            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Displaying error: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

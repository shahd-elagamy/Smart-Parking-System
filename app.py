from flask import Flask, render_template, jsonify, Response
import cv2
import json
from parking import ParkingManager

app = Flask(__name__)
open("occupied_spots_log.txt", "w").close()
manager = ParkingManager(model_path="yolo11s.pt", json_file="bounding_boxes.json")
cap = cv2.VideoCapture("parking.mp4")
print("Video opened:", cap.isOpened())

# وظيفة لقراءة البيانات من الملف وتحديث حالة الأماكن المشغولة
def read_occupied_spots():
    try:
        with open("occupied_spots_log.txt", "r") as file:
            lines = file.readlines()
        spots = []
        occupied_spots_set = set()  # Set لتخزين الأماكن المشغولة بشكل فريد
        for line in lines:
            if "OCCUPIED" in line:
                # استخراج البيانات من السطر
                timestamp, spot, coords = line.split(" - ")
                spot_number = int(spot.split()[1])  # رقم المكان المشغول
                coords = json.loads(coords.replace("Coords:", ""))  # تحويل الإحداثيات
                if spot_number not in occupied_spots_set:  # التحقق من التكرار
                    occupied_spots_set.add(spot_number)  # إضافة المكان إلى set
                    spots.append({"timestamp": timestamp, "spot": spot_number, "coords": coords})
        return spots
    except Exception as e:
        print("Error reading file:", e)
        return []

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
def get_data():
    try:
        with open("occupied_spots_log.txt", "r") as file:
            lines = file.readlines()

        # استخراج آخر سطر يحتوي على البيانات الكلية (Occupied/Available)
        last_summary_line = None
        for line in reversed(lines):
            if "Occupied:" in line and "Available:" in line:
                last_summary_line = line.strip()
                break

        if not last_summary_line:
            return jsonify({"error": "No summary data found"})

        # استخراج الأرقام من السطر
        import re
        match = re.search(r"Occupied:\s*(\d+)\s*-\s*Available:\s*(\d+)", last_summary_line)
        if match:
            occupied_spots = int(match.group(1))
            available_spots = int(match.group(2))
            total_spots = occupied_spots + available_spots
        else:
            return jsonify({"error": "Summary format invalid"})

        # استخراج الأماكن المشغولة
        spots = []
        occupied_spots_set = set()
        for line in reversed(lines):
            if "OCCUPIED" in line:
                parts = line.split(" - ")
                if len(parts) >= 4:
                    timestamp = parts[0]
                    spot_number = int(parts[1].split()[1])
                    coords = json.loads(parts[2].replace("Coords:", "").strip())

                    if spot_number not in occupied_spots_set:
                        occupied_spots_set.add(spot_number)
                        spots.append({
                            "timestamp": timestamp,
                            "spot": spot_number,
                            "coords": coords
                        })

            if len(spots) >= occupied_spots:
                break

        return jsonify({
            "occupied": occupied_spots,
            "available": available_spots,
            "total": total_spots,
            "spots": spots
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ✅ Route لإرسال الفيديو الحي
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (1080, 600))
        frame, _ = manager.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

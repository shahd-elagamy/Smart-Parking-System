import cv2
from parking import ParkingManager

# Video capture
cap = cv2.VideoCapture("parking.mp4")

# Initialize parking management object
parking_manager = ParkingManager(
    model_path="yolo11s.pt",  # path to model file
    json_file="bounding_boxes.json"  # path to parking annotations file
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame (optional)
    resized_frame = cv2.resize(frame, (1080, 600))
    
    # Call the correct method name (process_frame instead of process_data)
    processed_frame, occupancy_info = parking_manager.process_frame(resized_frame)
    
    # Display the result
    cv2.imshow("Parking Monitoring", processed_frame)
    
    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import os

# Fix OpenMP library issue (for Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv8 model (lightweight version)
model = YOLO("yolov8n.pt")

# --- SETTINGS ---
video_path = "traffic.mp4"   # your video file
frame_limit = 1800            # roughly 1 minute (30 FPS × 60 sec)
show_labels = True

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video opened at {fps:.1f} FPS")

frame_count = 0
vehicle_count = 0
line_y = 300  # horizontal line position (adjust based on your video)
offset = 6    # margin of error for line crossing
centers_tracked = set()

while True:
    ret, frame = cap.read()
    if not ret or frame_count > frame_limit:
        break
    frame_count += 1

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw the counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    # Draw boxes for vehicles only
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Draw rectangle and center point
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                if show_labels:
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

                # Check if vehicle crosses the line
                if (line_y - offset) < cy < (line_y + offset):
                    # Make a unique ID for this vehicle using its center position
                    vehicle_id = (cx // 10, cy // 10)
                    if vehicle_id not in centers_tracked:
                        vehicle_count += 1
                        centers_tracked.add(vehicle_id)

    # Display count
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("YOLO Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Detection finished — Total vehicles counted: {vehicle_count}")

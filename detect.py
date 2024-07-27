from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("best.pt")

# Predict using the default camera (source=0)
model.predict(source="Test_night1.mp4", show=True, save=True)

# Wait for a key press and then exit
cv2.waitKey(1)
cv2.destroyAllWindows()

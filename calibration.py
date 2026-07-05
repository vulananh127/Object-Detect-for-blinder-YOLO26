# calibration.py — chạy 1 lần để lấy FOCAL_LENGTH
import cv2

DROIDCAM_URL   = "http://192.168.71.104:4747/video"
KNOWN_DISTANCE = 100   # cm 
KNOWN_WIDTH    = 50    # cm 

cap = cv2.VideoCapture(DROIDCAM_URL)
ret, frame = cap.read()
cap.release()

if not ret:
    print(" Không kết nối được DroidCam!")
    exit()

print(f"Kéo chuột chọn vùng vật thể rộng {KNOWN_WIDTH}cm đang đặt cách {KNOWN_DISTANCE}cm")
roi = cv2.selectROI("Calibration", frame, fromCenter=False)
cv2.destroyAllWindows()

pixel_width = roi[2]
focal_length = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH

print(f"\n Pixel width : {pixel_width} px")
print(f"FOCAL_LENGTH = {focal_length:.2f}")
print(f"\n→ Copy giá trị này vào FOCAL_LENGTH trong file chính!")
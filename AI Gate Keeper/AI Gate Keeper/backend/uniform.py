import cv2
import numpy as np

# Open camera
cap = cv2.VideoCapture(0)

# HSV range for BLUE uniform (adjust if needed)
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # ---- Upper body region (shirt area) ----
    upper_body = frame[int(height*0.25):int(height*0.6), int(width*0.3):int(width*0.7)]

    # Convert to HSV
    hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)

    # Mask for uniform color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Percentage of uniform color
    uniform_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    uniform_ratio = uniform_pixels / total_pixels

    # Decision threshold
    if uniform_ratio > 0.25:
        status = "UNIFORM DETECTED"
        color = (0, 255, 0)
    else:
        status = "NO UNIFORM"
        color = (0, 0, 255)

    # Display
    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.rectangle(frame,
                  (int(width*0.3), int(height*0.25)),
                  (int(width*0.7), int(height*0.6)),
                  color, 2)

    cv2.imshow("Uniform Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

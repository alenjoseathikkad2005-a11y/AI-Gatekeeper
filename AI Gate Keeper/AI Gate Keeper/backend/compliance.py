import cv2
import numpy as np

class ComplianceDetector:

    def __init__(self):
        self.enabled = True

        # Blue uniform HSV range
        self.lower_blue = np.array([90, 40, 40])
        self.upper_blue = np.array([140, 255, 255])


    # ================= UNIFORM DETECTION =================
    def detect_uniform(self, frame):

        height, width, _ = frame.shape

        # Larger upper body area
        upper_body = frame[
            int(height * 0.25):int(height * 0.65),
            int(width * 0.2):int(width * 0.8)
        ]

        hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        uniform_pixels = cv2.countNonZero(mask)
        total_pixels = mask.size

        ratio = uniform_pixels / total_pixels

        # Lower threshold for better detection
        return ratio > 0.12


    # ================= ID CARD DETECTION =================
    def detect_id_card(self, frame):

        height, width, _ = frame.shape

        chest_area = frame[
            int(height * 0.35):int(height * 0.75),
            int(width * 0.35):int(width * 0.65)
        ]

        gray = cv2.cvtColor(chest_area, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 60, 160)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:

            x, y, w, h = cv2.boundingRect(c)

            area = w * h

            # More flexible detection
            if 1500 < area < 30000:

                aspect_ratio = w / float(h)

                if 0.4 < aspect_ratio < 2.5:
                    return True

        return False


    # ================= FRAME CHECK =================
    def check_frame(self, frame):

        uniform_ok = self.detect_uniform(frame)
        id_card_ok = self.detect_id_card(frame)

        compliant = uniform_ok and id_card_ok

        return {
            "enabled": True,
            "uniform_ok": uniform_ok,
            "id_card_ok": id_card_ok,
            "compliant": compliant
        }


compliance_detector = ComplianceDetector()
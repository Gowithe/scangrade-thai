# utils.py
import cv2
import numpy as np
import smtplib
import os
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# =========================
# Email config
# =========================
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", SMTP_USER)
ADMIN_KEY = os.getenv("ADMIN_KEY")

# =========================
# Warp config
# =========================
TARGET_WIDTH = 1600
TARGET_HEIGHT = 2300

AUTO_CROP_TOP = 0
AUTO_CROP_BOTTOM = 40
AUTO_CROP_LEFT = 0
AUTO_CROP_RIGHT = 0


# =========================
# Perspective helpers
# =========================
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype="float32")


def warp_from_four_points(image, pts):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (TARGET_WIDTH, TARGET_HEIGHT))


def auto_detect_and_warp(image_bgr):
    if image_bgr is None:
        return None

    orig = image_bgr.copy()
    h, w = orig.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale = 1000.0 / max(w, h)
    resized = cv2.resize(orig, (int(w * scale), int(h * scale))) if scale < 1 else orig.copy()
    scale = min(scale, 1.0)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = resized.shape[0] * resized.shape[1]

    for min_ratio in (0.10, 0.05, 0.02):
        for cnt in contours:
            if cv2.contourArea(cnt) < img_area * min_ratio:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                warped = warp_from_four_points(orig, approx.reshape(4, 2) / scale)
                hf, wf = warped.shape[:2]
                crop = warped[
                    AUTO_CROP_TOP:hf - AUTO_CROP_BOTTOM,
                    AUTO_CROP_LEFT:wf - AUTO_CROP_RIGHT
                ]
                return cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
    return None


# =========================
# Answer key normalizer
# =========================
def normalize_answer_key_str(s, num_questions):
    if not s:
        return ""
    s = str(s).upper()
    clean = "".join(ch for ch in s if ch in "ABCDE")
    try:
        n = int(num_questions)
        return clean[:n]
    except Exception:
        return clean


# =========================
# Email helpers
# =========================
def send_login_otp_email(to_email, code, ttl_min):
    if not SMTP_USER or not to_email:
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = "ScanGrade - OTP"

    body = f"OTP Code: {code}\nExpires in {ttl_min} minutes."
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            if SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# app.py
from flask import Flask, render_template, request, redirect, session, send_from_directory, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
import cv2
import numpy as np
import base64
import os
import random
import uuid
import time
import requests
import re
from dotenv import load_dotenv

# Import modules
import db
import utils
import omr60
import omr80

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "SECRET_KEY_MISSING")

# -------------------------
# Upload safety (ข้อ 1)
# -------------------------
# จำกัดขนาดไฟล์อัปโหลดทั้งหมด (sheet + slip)
# default 8MB (ปรับได้ผ่าน env MAX_UPLOAD_MB)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    max_mb = int(os.getenv("MAX_UPLOAD_MB", "8"))
    return (
        "<h3>ไฟล์ใหญ่เกินไป</h3>"
        f"<p>กรุณาอัปโหลดรูปขนาดไม่เกิน {max_mb}MB</p>"
        "<p>แนะนำ: ปิดโหมด 48MP / เลือกภาพขนาดปกติ / ส่งเป็น JPEG</p>"
        "<a href='/'>กลับหน้าหลัก</a>",
        413,
    )


def is_allowed_image_filename(filename: str) -> bool:
    if not filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_IMAGE_EXTS


def read_image_from_filestorage(file_storage):
    """
    อ่านรูปจาก Flask FileStorage แบบปลอดภัย
    คืนค่า: img (BGR) หรือ None
    """
    try:
        data = file_storage.read()
        file_storage.stream.seek(0)  # reset pointer เผื่อใช้ต่อ
        if not data:
            return None
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def downscale_image(img, max_side=2000):
    """
    ลดขนาดรูปก่อนประมวลผล เพื่อให้เร็วขึ้น/นิ่งขึ้น
    """
    try:
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return img


# -------------------------
# Config
# -------------------------
COOKIE_SECURE = (os.getenv("COOKIE_SECURE", "0") == "1")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
DEBUG_SHOW_OTP = (os.getenv("DEBUG_SHOW_OTP", "0") == "1")

OTP_TTL = 300
PENDING_OTP = {}

# EasySlip
SLIP_API_KEY = os.getenv("SLIP_API_KEY", "")
EASYSLIP_VERIFY_URL = os.getenv("EASYSLIP_VERIFY_URL", "https://developer.easyslip.com/api/v1/verify")

# Init DB
db.init_db()


# -------------------------
# Helpers
# -------------------------
def cleanup_expired_otp():
    now = time.time()
    for k in list(PENDING_OTP.keys()):
        if PENDING_OTP[k]["exp"] < now:
            PENDING_OTP.pop(k, None)


def ensure_logged_in():
    username = session.get("username")
    if not username:
        return None, None, redirect("/login")
    user = db.get_user(username)
    if not user:
        session.pop("username", None)
        return None, None, redirect("/login")
    return username, user, None


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _last4_digits(s: str) -> str:
    digits = re.sub(r"\D", "", s or "")
    return digits[-4:] if len(digits) >= 4 else ""


def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9ก-๙]+", "", s)
    return s


# -------------------------
# Anti double-submit (ข้อ 2 / BACKEND)
# -------------------------
def start_action_lock(action_name: str, ttl_sec: int = 40) -> bool:
    """
    ล็อก action ต่อ session กันกดซ้ำ
    return False ถ้า action เดิมกำลังทำอยู่
    """
    now = time.time()
    locks = session.get("_action_locks", {})

    # cleanup expired
    locks = {k: v for k, v in locks.items() if v > now}

    if action_name in locks:
        session["_action_locks"] = locks
        return False

    locks[action_name] = now + ttl_sec
    session["_action_locks"] = locks
    return True


def end_action_lock(action_name: str):
    locks = session.get("_action_locks", {})
    locks.pop(action_name, None)
    session["_action_locks"] = locks


# -------------------------
# Minimal file cleanup (uploads/slips)
# -------------------------
def cleanup_old_files(folder: str, max_age_sec: int, min_age_sec: int = 180, exts=None, max_delete: int = 200):
    """
    ลบไฟล์ที่เก่ากว่า max_age_sec แต่จะไม่ลบไฟล์ที่อายุน้อยกว่า min_age_sec (กันลบไฟล์ที่เพิ่งสร้าง/กำลังใช้)
    - exts: set ของนามสกุลที่อนุญาตให้ลบ เช่น {".jpg",".jpeg",".png",".webp"}
    - max_delete: จำกัดจำนวนไฟล์ที่ลบต่อครั้ง กัน loop นาน
    """
    try:
        if not os.path.isdir(folder):
            return 0

        now = time.time()
        deleted = 0

        for fname in os.listdir(folder):
            if deleted >= max_delete:
                break

            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue

            if exts:
                ext = os.path.splitext(fname.lower())[1]
                if ext not in exts:
                    continue

            try:
                age = now - os.path.getmtime(path)
                if age < min_age_sec:
                    continue
                if age > max_age_sec:
                    os.remove(path)
                    deleted += 1
            except Exception:
                pass

        return deleted
    except Exception:
        return 0


def maybe_cleanup():
    """
    Minimal: รัน cleanup แบบสุ่ม 10% ของ request
    """
    try:
        if random.random() > 0.10:
            return

        os.makedirs("uploads", exist_ok=True)
        os.makedirs("slips", exist_ok=True)

        cleanup_old_files(
            folder="uploads",
            max_age_sec=int(os.getenv("UPLOADS_TTL_SEC", str(6 * 3600))),   # 6 ชม.
            min_age_sec=int(os.getenv("UPLOADS_MIN_AGE_SEC", "180")),      # 3 นาที
            exts={".jpg", ".jpeg", ".png", ".webp"},
            max_delete=200
        )

        cleanup_old_files(
            folder="slips",
            max_age_sec=int(os.getenv("SLIPS_TTL_SEC", str(72 * 3600))),    # 72 ชม.
            min_age_sec=int(os.getenv("SLIPS_MIN_AGE_SEC", "300")),        # 5 นาที
            exts={".jpg", ".jpeg", ".png", ".webp"},
            max_delete=200
        )

    except Exception:
        pass


# ✅ per-session manual upload helpers
def _ensure_upload_dir():
    os.makedirs("uploads", exist_ok=True)


def _new_manual_upload_path():
    _ensure_upload_dir()
    return os.path.join("uploads", f"sheet_{uuid.uuid4().hex}.jpg")


def normalize_slip_to_jpg(upload_file, max_width=1600, jpeg_quality=92):
    """
    แปลงไฟล์ที่อัปโหลดให้เป็น JPG มาตรฐาน (กัน invalid_image)
    คืนค่า: (ok, message, jpg_bytes)
    """
    raw = upload_file.read()
    upload_file.stream.seek(0)
    if not raw:
        return False, "ไฟล์ว่างเปล่า", None

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False, "ไฟล์ที่อัปโหลดไม่ใช่รูปที่รองรับ (แนะนำ: แคปหน้าจอสลิปแล้วบันทึกเป็น JPG/PNG)", None

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return False, "ไม่สามารถแปลงรูปเป็น JPG ได้", None

    return True, "ok", buf.tobytes()


def verify_slip_with_easyslip(file_path: str, expected_amount: float):
    """
    ตรวจสลิปกับ EasySlip
    คืนค่า: (is_valid, message, slip_ref, paid_amount)
    """
    if not SLIP_API_KEY:
        return False, "SLIP_API_KEY ไม่ถูกตั้งค่าใน .env", None, None

    if not os.path.exists(file_path):
        return False, "ไม่พบไฟล์สลิป", None, None

    try:
        with open(file_path, "rb") as f:
            headers = {"Authorization": f"Bearer {SLIP_API_KEY}"}
            files = {"file": ("slip.jpg", f, "image/jpeg")}
            r = requests.post(EASYSLIP_VERIFY_URL, headers=headers, files=files, timeout=25)
    except Exception as e:
        return False, f"เชื่อมต่อ EasySlip ไม่ได้: {e}", None, None

    if r.status_code != 200:
        try:
            js_err = r.json()
            msg = js_err.get("message") or js_err.get("error") or str(js_err)
        except Exception:
            msg = (r.text[:500] if r.text else "ไม่พบรายละเอียด error")
        return False, f"EasySlip HTTP {r.status_code}: {msg}", None, None

    try:
        js = r.json()
    except Exception:
        return False, "EasySlip ตอบกลับไม่ใช่ JSON", None, None

    data = js.get("data") or {}

    slip_ref = (
        data.get("transRef")
        or data.get("transRefId")
        or data.get("transactionId")
        or data.get("transaction_id")
        or data.get("ref")
        or data.get("reference")
        or data.get("slipRef")
    )
    if not slip_ref:
        return False, "ตรวจสลิปได้ แต่ไม่พบรหัสธุรกรรม (transRef) เพื่อกันสลิปซ้ำ", None, None

    paid_amount = None
    amt_obj = data.get("amount")
    if isinstance(amt_obj, dict):
        paid_amount = _safe_float(amt_obj.get("amount"))
    if paid_amount is None:
        paid_amount = (
            _safe_float(data.get("amount"))
            or _safe_float(data.get("paid_amount"))
            or _safe_float(data.get("total"))
        )

    if paid_amount is None:
        return False, "ตรวจสลิปได้ แต่ไม่พบจำนวนเงินในข้อมูลสลิป", slip_ref, None

    if float(paid_amount) != float(expected_amount):
        return False, f"ยอดเงินไม่ตรงแพ็ก (สลิป {paid_amount} บาท, ต้องเป็น {expected_amount} บาท)", slip_ref, paid_amount

    expected_name_th = (os.getenv("EXPECTED_RECEIVER_NAME_TH") or "").strip()
    expected_name_en = (os.getenv("EXPECTED_RECEIVER_NAME_EN") or "").strip()
    expected_bank_last4 = (os.getenv("EXPECTED_BANK_ACCOUNT_LAST4") or "").strip()
    expected_proxy_last4 = (os.getenv("EXPECTED_PROXY_LAST4") or "").strip()

    receiver = data.get("receiver") or {}
    acc = receiver.get("account") or {}

    name_obj = acc.get("name") or {}
    receiver_name_th = ""
    receiver_name_en = ""
    if isinstance(name_obj, dict):
        receiver_name_th = (name_obj.get("th") or "").strip()
        receiver_name_en = (name_obj.get("en") or "").strip()
    else:
        receiver_name_th = str(name_obj).strip()

    bank_obj = acc.get("bank") or {}
    proxy_obj = acc.get("proxy") or {}

    receiver_bank_account = str(bank_obj.get("account") or "").strip()
    receiver_proxy_account = str(proxy_obj.get("account") or "").strip()

    got_bank_last4 = _last4_digits(receiver_bank_account)
    got_proxy_last4 = _last4_digits(receiver_proxy_account)

    checks_enabled = any([expected_name_th, expected_name_en, expected_bank_last4, expected_proxy_last4])
    if checks_enabled:
        ok_dest = False

        if expected_name_th and _norm_name(expected_name_th) in _norm_name(receiver_name_th):
            ok_dest = True
        if (not ok_dest) and expected_name_en and _norm_name(expected_name_en) in _norm_name(receiver_name_en):
            ok_dest = True
        if (not ok_dest) and expected_bank_last4 and got_bank_last4 and got_bank_last4 == expected_bank_last4:
            ok_dest = True
        if (not ok_dest) and expected_proxy_last4 and got_proxy_last4 and got_proxy_last4 == expected_proxy_last4:
            ok_dest = True

        if not ok_dest:
            return False, "ผู้รับเงินไม่ตรง (ไม่ใช่บัญชี/PromptPay ร้าน)", slip_ref, paid_amount

    return True, "สลิปถูกต้อง", slip_ref, paid_amount


# -------------------------
# Routes
# -------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    cleanup_expired_otp()
    error = None

    device_id = request.cookies.get("device_id")
    if not device_id:
        device_id = str(uuid.uuid4())
    db.upsert_device(device_id)

    current_user = session.get("username")
    current_credits = None
    if current_user:
        u = db.get_user(current_user)
        current_credits = u["credits"] if u else None

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip().lower()
        if not identifier or "@" not in identifier:
            error = "Invalid Email"
        else:
            code = f"{random.randint(0, 999999):06d}"
            token = str(uuid.uuid4())
            PENDING_OTP[token] = {"identifier": identifier, "code": code, "exp": time.time() + OTP_TTL}

            if utils.send_login_otp_email(identifier, code, OTP_TTL // 60):
                resp = redirect(f"/verify?token={token}")
                resp.set_cookie(
                    "device_id",
                    device_id,
                    max_age=31536000,
                    httponly=True,
                    secure=COOKIE_SECURE,
                    samesite="Lax",
                )
                return resp
            else:
                error = "Cannot send Email"

    resp = app.make_response(
        render_template("login.html", error=error, current_user=current_user, current_credits=current_credits)
    )
    resp.set_cookie("device_id", device_id, max_age=31536000, httponly=True, secure=COOKIE_SECURE, samesite="Lax")
    return resp


@app.route("/verify", methods=["GET", "POST"])
def verify():
    cleanup_expired_otp()
    token = request.args.get("token") or request.form.get("token")
    if not token or token not in PENDING_OTP:
        return redirect("/login")

    data = PENDING_OTP.get(token)
    identifier = data["identifier"]
    error = None

    device_id = request.cookies.get("device_id") or str(uuid.uuid4())

    if request.method == "POST":
        otp_input = request.form.get("otp", "").strip()
        if time.time() > data["exp"]:
            error = "OTP Expired"
        elif otp_input != data["code"]:
            error = "Wrong OTP"
        else:
            PENDING_OTP.pop(token, None)
            session["username"] = identifier

            user = db.get_user(identifier)
            device = db.get_device(device_id)
            if not device:
                db.upsert_device(device_id)

            if not user:
                device = db.get_device(device_id)
                free_credits = 20 if not device["used_free"] else 0
                db.create_user(identifier, free_credits)
                if free_credits > 0:
                    db.mark_device_used_free(device_id)

            resp = redirect("/")
            resp.set_cookie(
                "device_id",
                device_id,
                max_age=31536000,
                httponly=True,
                secure=COOKIE_SECURE,
                samesite="Lax",
            )
            return resp

    dev_otp = f"Debug OTP: {data['code']}" if DEBUG_SHOW_OTP else None
    return render_template("verify.html", identifier=identifier, token=token, error=error, dev_otp_message=dev_otp)


@app.route("/logout")
def logout():
    session.pop("username", None)

    # cleanup manual file (ถ้าค้างอยู่)
    manual_path = session.pop("manual_upload_path", None)
    try:
        if manual_path and os.path.exists(manual_path):
            os.remove(manual_path)
    except Exception:
        pass

    return redirect("/login")


@app.route("/")
def upload_page():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp

    num_questions = int(request.args.get("num_questions", "60"))
    subject = (request.args.get("subject") or "").strip()

    warp_fail_message = session.pop("warp_fail_message", "")

    last_key = session.pop("last_answer_key", "")
    last_subject = session.pop("last_subject", "")
    last_numq = session.pop("last_num_questions", None)

    if last_numq is not None:
        try:
            num_questions = int(last_numq)
        except Exception:
            pass

    if last_subject:
        subject = last_subject

    answer_key_str = ""
    selected_subject = ""

    if subject:
        answer_key_str = db.get_saved_key(username, subject, num_questions) or ""
        selected_subject = subject
    else:
        subjects = db.list_saved_subjects(username, num_questions)
        if subjects:
            selected_subject = subjects[0]["subject"]
            answer_key_str = db.get_saved_key(username, selected_subject, num_questions) or ""

    if last_key:
        answer_key_str = last_key

    subjects = db.list_saved_subjects(username, num_questions)

    return render_template(
        "upload.html",
        num_questions=num_questions,
        answer_key_str=answer_key_str,
        username=username,
        credits=user["credits"],
        subjects=subjects,
        selected_subject=selected_subject,
        warp_fail_message=warp_fail_message,
    )


@app.route("/save_key", methods=["POST"])
def save_key():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp

    subject = (request.form.get("subject") or "").strip()
    num_questions = int(request.form.get("num_questions", "60"))
    key_str = (request.form.get("answer_key") or "").strip()

    key_str = utils.normalize_answer_key_str(key_str, num_questions)

    if not subject:
        return "กรุณากรอกชื่อวิชา/ชุดข้อสอบ", 400
    if not key_str:
        return "กรุณากรอกเฉลยก่อนบันทึก", 400

    db.upsert_saved_key(username, subject, num_questions, key_str)
    return redirect(f"/?num_questions={num_questions}&subject={subject}")


@app.route("/api/subjects")
def api_subjects():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp
    num_questions = int(request.args.get("num_questions", "60"))
    items = db.list_saved_subjects(username, num_questions)
    return jsonify({"ok": True, "items": items})


@app.route("/api/answer_key")
def api_answer_key():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp

    subject = (request.args.get("subject") or "").strip()
    num_questions = int(request.args.get("num_questions", "60"))

    if not subject:
        return jsonify({"ok": False, "message": "missing subject"}), 400

    key_str = db.get_saved_key(username, subject, num_questions) or ""
    return jsonify({"ok": True, "subject": subject, "num_questions": num_questions, "key_str": key_str})


@app.route("/buy", methods=["GET", "POST"])
def buy_credits():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp

    # ✅ minimal cleanup (สุ่ม 10%)
    maybe_cleanup()

    os.makedirs("slips", exist_ok=True)

    if request.method == "POST":
        # ✅ กันกดซ้ำเติมเงิน
        if not start_action_lock("buy_credits", ttl_sec=60):
            return "⏳ กำลังตรวจสอบการชำระเงิน กรุณารอสักครู่ (อย่ากดซ้ำ)", 429

        try:
            pkg = request.form.get("package")
            slip = request.files.get("slip")

            if pkg not in db.PACKAGES:
                return "แพ็กเกจไม่ถูกต้อง", 400
            if not slip or slip.filename == "":
                return "กรุณาอัปโหลดสลิป", 400
            if not is_allowed_image_filename(slip.filename):
                return "รองรับสลิปเป็นรูป .jpg .jpeg .png .webp เท่านั้น", 400

            expected_price = float(db.PACKAGES[pkg]["price"])

            ok, msg, jpg_bytes = normalize_slip_to_jpg(slip)
            if not ok:
                return msg, 400

            safe_name = f"slip_{int(time.time())}_{random.randint(100000,999999)}.jpg"
            save_path = os.path.join("slips", safe_name)
            with open(save_path, "wb") as f:
                f.write(jpg_bytes)

            is_valid, verify_msg, slip_ref, paid_amount = verify_slip_with_easyslip(
                save_path, expected_amount=expected_price
            )

            # UI defaults
            status_title = "⏳ กำลังตรวจสอบ..."
            status_color = "#f59e0b"
            status_icon = "⏳"
            btn_text = "กลับหน้าหลัก"
            btn_link = "/"
            status_desc = "ระบบกำลังตรวจสอบรายการ..."

            # กันซ้ำด้วย slip_ref
            if is_valid and db.is_slip_ref_used(slip_ref):
                is_valid = False
                verify_msg = "สลิปนี้ถูกใช้ไปแล้ว (ตรวจพบธุรกรรมซ้ำ)"

            order_id = None
            pkg_info = db.PACKAGES[pkg]

            if is_valid:
                try:
                    order_id = db.create_order(username, pkg, safe_name, slip_ref)  # <-- int
                except ValueError:
                    is_valid = False
                    verify_msg = "สลิปนี้ถูกใช้ไปแล้ว (ระบบป้องกันการใช้ซ้ำ)"

            if is_valid and order_id:
                print(f"[AUTO-APPROVE] Order #{order_id} Verified! slip_ref={slip_ref}")

                order_row, user_row = db.approve_order_and_add_credits(order_id)
                if not user_row:
                    is_valid = False
                    verify_msg = "เติมเครดิตไม่สำเร็จ (ไม่พบผู้ใช้ในระบบ)"
                else:
                    new_credits = user_row["credits"]
                    status_title = "✅ เติมเครดิตสำเร็จ!"
                    status_color = "#10b981"
                    status_icon = "🎉"
                    status_desc = (
                        f"ระบบตรวจสอบเรียบร้อย<br>"
                        f"ยอดเงิน: <b>{paid_amount}</b> บาท<br>"
                        f"คุณได้รับเครดิตเพิ่ม <b>{pkg_info['credits']}</b> ครั้ง<br>"
                        f"เครดิตรวม: <b>{new_credits}</b><br>"
                        f"<span style='font-size:0.85rem;color:#94a3b8;'>ref: {slip_ref}</span>"
                    )
                    btn_text = "กลับหน้าหลัก"
                    btn_link = "/"

            if not is_valid:
                print(f"[SLIP-FAIL] pkg={pkg} user={username} -> {verify_msg} ref={slip_ref}")
                try:
                    os.remove(save_path)
                except Exception:
                    pass

                status_title = "❌ ชำระเงินไม่สำเร็จ"
                status_color = "#ef4444"
                status_icon = "⚠️"
                status_desc = f"""
                <b>เกิดข้อผิดพลาด:</b><br>
                <span style="color:#fca5a5;">{verify_msg}</span><br><br>
                กรุณาตรวจสอบความถูกต้องแล้วลองใหม่อีกครั้ง
                """
                btn_text = "ลองใหม่อีกครั้ง"
                btn_link = "/buy"

            show_home_secondary = (btn_link != "/")

            html = f"""
            <!doctype html>
            <html lang="th">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;800&display=swap" rel="stylesheet">
              <style>
                * {{ box-sizing: border-box; }}
                body {{
                  font-family: 'Sarabun', sans-serif;
                  background: #0f172a; color: #e5e7eb;
                  margin: 0; padding: 20px;
                  display: flex; align-items: center; justify-content: center;
                  min-height: 100vh;
                }}
                .card {{
                  background: rgba(30, 41, 59, 0.75);
                  backdrop-filter: blur(16px);
                  padding: 40px 30px;
                  border-radius: 24px;
                  text-align: center;
                  width: 100%; max-width: 420px;
                  border: 1px solid rgba(255, 255, 255, 0.08);
                  box-shadow: 0 20px 40px rgba(0,0,0,0.3);
                }}
                h2 {{ color: {status_color}; margin-top: 0; font-size: 1.5rem; margin-bottom: 16px; }}
                p {{ font-size: 1.05rem; color: #cbd5e1; margin-bottom: 24px; line-height: 1.6; }}
                .btn-row{{ display:flex; gap:10px; flex-direction:column; }}
                .btn {{
                  display: block; width: 100%; padding: 14px 0;
                  background: linear-gradient(135deg, #3b82f6, #2563eb);
                  color: white; text-decoration: none; font-weight: 700;
                  border-radius: 12px; font-size: 1rem;
                  box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
                }}
                .btn-retry {{
                  background: linear-gradient(135deg, #ef4444, #b91c1c);
                  box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
                }}
                .btn-secondary {{
                  background: rgba(148,163,184,0.15);
                  box-shadow: none;
                  border: 1px solid rgba(255,255,255,0.12);
                }}
              </style>
            </head>
            <body>
              <div class="card">
                <div style="font-size: 4rem; margin-bottom: 20px;">{status_icon}</div>
                <h2>{status_title}</h2>
                <p>{status_desc}</p>

                <div class="btn-row">
                  <a href="{btn_link}" class="btn {"btn-retry" if btn_link=="/buy" else ""}">{btn_text}</a>
                  {"<a href='/' class='btn btn-secondary'>กลับหน้าหลัก</a>" if show_home_secondary else ""}
                </div>
              </div>
            </body>
            </html>
            """
            return html

        finally:
            end_action_lock("buy_credits")

    # GET
    return render_template(
        "buy.html",
        username=username,
        credits=user["credits"],
        packages=db.PACKAGES,
        payment_name=os.getenv("PAYMENT_NAME", ""),
        payment_promptpay=os.getenv("PAYMENT_PROMPTPAY", ""),
        payment_note=os.getenv("PAYMENT_NOTE", ""),
        payment_qr_image=os.getenv("PAYMENT_QR_IMAGE", ""),
    )


@app.route("/auto_grade", methods=["POST"])
def auto_grade():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp
    if user["credits"] <= 0:
        return redirect("/buy")

    # ✅ minimal cleanup (สุ่ม 10%)
    maybe_cleanup()

    # ✅ กันกดซ้ำ
    if not start_action_lock("auto_grade", ttl_sec=45):
        session["warp_fail_message"] = "⏳ ระบบกำลังตรวจอยู่ กรุณารอสักครู่ (อย่ากดซ้ำ)"
        return redirect("/")

    try:
        file = request.files.get("sheet")
        if not file or file.filename == "":
            session["warp_fail_message"] = "❌ กรุณาเลือกรูปกระดาษคำตอบก่อน"
            return redirect("/")

        if not is_allowed_image_filename(file.filename):
            session["warp_fail_message"] = "❌ รองรับเฉพาะไฟล์รูป .jpg .jpeg .png .webp"
            return redirect("/")

        num_questions = int(request.form.get("num_questions", "60"))
        key_str = (request.form.get("answer_key") or "").strip()
        subject = (request.form.get("subject") or "").strip()

        session["last_answer_key"] = utils.normalize_answer_key_str(key_str, num_questions)
        session["last_subject"] = subject
        session["last_num_questions"] = num_questions

        img = read_image_from_filestorage(file)
        if img is None:
            session["warp_fail_message"] = "❌ ไม่สามารถอ่านไฟล์รูปได้ (ไฟล์เสียหรือไม่รองรับ) กรุณาลองถ่าย/เลือกใหม่"
            return redirect(f"/?num_questions={num_questions}&subject={subject}" if subject else f"/?num_questions={num_questions}")

        img = downscale_image(img, max_side=2000)

        warped = utils.auto_detect_and_warp(img)
        if warped is None:
            session["warp_fail_message"] = (
                "❌ ระบบไม่สามารถตรวจจับมุมกระดาษคำตอบได้<br><br>"
                "💡 คำแนะนำ:<br>"
                "- ถ่ายในที่แสงสว่างเพียงพอ<br>"
                "- ให้เห็นกระดาษทั้ง 4 มุมชัดเจน<br>"
                "- วางกระดาษบนพื้นหลังเรียบ/ตัดของรกออก<br><br>"
                "หรือเลือก <b>โหมด Manual</b> เพื่อกำหนดมุมเอง"
            )
            return redirect(f"/?num_questions={num_questions}&subject={subject}" if subject else f"/?num_questions={num_questions}")

        try:
            omr = omr60 if num_questions == 60 else omr80
            answers, eff_key, detail, stats, debug_img = omr.process_auto(warped, key_str)
        except Exception as e:
            session["warp_fail_message"] = f"❌ ตรวจไม่สำเร็จ: {e}"
            return redirect(f"/?num_questions={num_questions}&subject={subject}" if subject else f"/?num_questions={num_questions}")

        db.set_user_credits(username, user["credits"] - 1)

        _, buf = cv2.imencode(".jpg", debug_img)
        debug_b64 = base64.b64encode(buf).decode("utf-8")

        return render_template(
            "result.html",
            answers=answers,
            stats=stats,
            detail=detail,
            debug_image=debug_b64,
            num_questions=num_questions,
            answer_key=eff_key,
            answer_key_str_raw=utils.normalize_answer_key_str(key_str, num_questions),
            username=username,
            credits=user["credits"] - 1,
        )

    finally:
        end_action_lock("auto_grade")


@app.route("/select", methods=["POST"])
def select_corners():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp

    # ✅ minimal cleanup (สุ่ม 10%)
    maybe_cleanup()

    file = request.files.get("sheet")
    if not file or file.filename == "":
        session["warp_fail_message"] = "❌ กรุณาเลือกรูปกระดาษคำตอบก่อน"
        return redirect("/")

    if not is_allowed_image_filename(file.filename):
        session["warp_fail_message"] = "❌ รองรับเฉพาะไฟล์รูป .jpg .jpeg .png .webp"
        return redirect("/")

    num_questions = int(request.form.get("num_questions", "60"))
    key_str = (request.form.get("answer_key") or "").strip()
    subject = (request.form.get("subject") or "").strip()

    session["last_answer_key"] = utils.normalize_answer_key_str(key_str, num_questions)
    session["last_subject"] = subject
    session["last_num_questions"] = num_questions

    img = read_image_from_filestorage(file)
    if img is None:
        session["warp_fail_message"] = "❌ ไม่สามารถอ่านไฟล์รูปได้ กรุณาลองถ่าย/เลือกใหม่"
        return redirect(f"/?num_questions={num_questions}&subject={subject}" if subject else f"/?num_questions={num_questions}")

    img = downscale_image(img, max_side=2400)

    # ✅ per-session manual image path (ไม่ชนกัน)
    old_path = session.get("manual_upload_path")
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except Exception:
            pass

    save_path = _new_manual_upload_path()
    cv2.imwrite(save_path, img)
    session["manual_upload_path"] = save_path

    _, buf = cv2.imencode(".jpg", img)
    return render_template(
        "select.html",
        img_data=base64.b64encode(buf).decode("utf-8"),
        answer_key_str=key_str,
        num_questions=num_questions,
        credits=user["credits"],
    )


@app.route("/grade", methods=["POST"])
def grade():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp
    if user["credits"] <= 0:
        return redirect("/buy")

    # ✅ minimal cleanup (สุ่ม 10%)
    maybe_cleanup()

    # ✅ กันกดซ้ำ (manual)
    if not start_action_lock("manual_grade", ttl_sec=45):
        session["warp_fail_message"] = "⏳ ระบบกำลังตรวจอยู่ กรุณารอสักครู่ (อย่ากดซ้ำ)"
        return redirect("/")

    try:
        manual_path = session.get("manual_upload_path")
        if not manual_path or not os.path.exists(manual_path):
            session["warp_fail_message"] = "❌ ไม่พบรูปสำหรับ Manual (กรุณาอัปโหลดใหม่)"
            return redirect("/")

        points_str = request.form.get("points", "")
        if not points_str:
            session["warp_fail_message"] = "❌ กรุณาเลือก 4 มุมก่อน"
            return redirect("/")

        try:
            pts = [list(map(float, p.split(","))) for p in points_str.split(";")]
            if len(pts) != 4:
                raise ValueError("points must be 4")
        except Exception:
            session["warp_fail_message"] = "❌ จุดมุมไม่ถูกต้อง กรุณาลองใหม่"
            return redirect("/")

        img = cv2.imread(manual_path)
        if img is None:
            session["warp_fail_message"] = "❌ ไม่สามารถอ่านรูปสำหรับ Manual ได้ กรุณาอัปโหลดใหม่"
            return redirect("/")

        warped = utils.warp_from_four_points(img, pts)

        num_questions = int(request.form.get("num_questions", "60"))
        key_str = (request.form.get("answer_key") or "").strip()

        try:
            omr = omr60 if num_questions == 60 else omr80
            answers, eff_key, detail, stats, debug_img = omr.process_auto(warped, key_str)
        except Exception as e:
            session["warp_fail_message"] = f"❌ ตรวจไม่สำเร็จ: {e}"
            return redirect(f"/?num_questions={num_questions}")

        db.set_user_credits(username, user["credits"] - 1)

        # ✅ cleanup manual file after success
        try:
            if manual_path and os.path.exists(manual_path):
                os.remove(manual_path)
        except Exception:
            pass
        session.pop("manual_upload_path", None)

        _, buf = cv2.imencode(".jpg", debug_img)
        return render_template(
            "result.html",
            answers=answers,
            stats=stats,
            detail=detail,
            debug_image=base64.b64encode(buf).decode("utf-8"),
            num_questions=num_questions,
            answer_key=eff_key,
            answer_key_str_raw=utils.normalize_answer_key_str(key_str, num_questions),
            username=username,
            credits=user["credits"] - 1,
        )

    finally:
        end_action_lock("manual_grade")


@app.route("/slips/<path:filename>")
def slip_file(filename):
    return send_from_directory("slips", filename)


@app.route("/admin/orders")
def admin_orders():
    if request.args.get("key") != ADMIN_KEY:
        return "Unauthorized", 403
    orders = db.get_all_orders()
    return render_template("admin_orders.html", orders=orders, admin_key=ADMIN_KEY)


@app.route("/admin/approve/<int:order_id>")
def admin_approve(order_id):
    if request.args.get("key") != ADMIN_KEY:
        return "Unauthorized", 403
    order_row, user_row = db.approve_order_and_add_credits(order_id)
    if not order_row:
        return "Not Found", 404
    if not user_row:
        return f"Approved (or already approved) but user not found for order #{order_id}", 200
    return f"Approved! User {user_row['username']} now has {user_row['credits']} credits."


@app.route("/next", methods=["POST"])
def next_sheet():
    username, user, resp = ensure_logged_in()
    if resp:
        return resp

    num_questions = int(request.form.get("num_questions", "60"))
    answer_key_str = (request.form.get("answer_key") or "").strip()
    subject = (request.form.get("subject") or "").strip()

    session["last_answer_key"] = utils.normalize_answer_key_str(answer_key_str, num_questions)
    session["last_subject"] = subject
    session["last_num_questions"] = num_questions

    return redirect(f"/?num_questions={num_questions}&subject={subject}" if subject else f"/?num_questions={num_questions}")


if __name__ == "__main__":
    # Production: รันด้วย gunicorn แทน (เช่น gunicorn app:app)
    app.run(host="0.0.0.0", port=5000, debug=False)

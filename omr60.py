# omr60.py
import cv2
import json
import numpy as np
import os

from model_loader import get_model

# =====================================
# CONFIG 60 ข้อ
# =====================================

MODEL_PATH = "runs/detect/train_AE52/weights/bestX.pt"
TEMPLATE_FILE = "questions_60.json"

OPTIONS = ["A", "B", "C", "D", "E"]
CONF_THRES = 0.10
MAX_SLOT_DIST = 150.0
NUM_QUESTIONS = 60

# ถ้าไม่กรอกเฉลย จะใช้ตัวนี้แทน
ANSWER_KEY_DEFAULT = {
    # 1: "A",
}

# ✅ shared model (โหลดครั้งเดียว)
model = get_model(MODEL_PATH)

# ✅ cache template+mapping (โหลดครั้งเดียว)
_TEMPLATE_CACHE = None
_MAPPING_CACHE = None

# =====================================
# TEMPLATE & MAPPING
# =====================================

def load_template():
    """โหลด template json -> dict {slot_index(int): (x,y)}"""
    json_path = TEMPLATE_FILE
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ template: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_slots = {int(k): (v["x"], v["y"]) for k, v in raw.items()}
    return all_slots

def build_slot_mapping(all_slots: dict):
    """
    slot_index -> (ข้อ, ตัวเลือก A–E)
    ตามลำดับ 1A..1E, 2A..2E ... (อิง sorted slot index)
    """
    mapping = {}
    sorted_indices = sorted(all_slots.keys())

    for pos, idx in enumerate(sorted_indices):
        qnum  = pos // len(OPTIONS) + 1
        opt_i = pos % len(OPTIONS)
        if 1 <= qnum <= NUM_QUESTIONS:
            mapping[idx] = (qnum, OPTIONS[opt_i])
    return mapping

def get_template_and_mapping():
    """✅ cache: โหลดครั้งเดียวต่อ process"""
    global _TEMPLATE_CACHE, _MAPPING_CACHE
    if _TEMPLATE_CACHE is None or _MAPPING_CACHE is None:
        _TEMPLATE_CACHE = load_template()
        _MAPPING_CACHE = build_slot_mapping(_TEMPLATE_CACHE)
        print(f"[60Q] Template cached: {TEMPLATE_FILE} | slots={len(_TEMPLATE_CACHE)} mapping={len(_MAPPING_CACHE)}")
    return _TEMPLATE_CACHE, _MAPPING_CACHE

# =====================================
# GEOMETRY HELPERS
# =====================================

def find_nearest_slot(xc, yc, all_slots, max_dist: float = MAX_SLOT_DIST):
    best_idx = None
    best_d2 = max_dist * max_dist

    for idx, (sx, sy) in all_slots.items():
        dx = xc - sx
        dy = yc - sy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_idx = idx

    return best_idx

# =====================================
# YOLO → ANSWERS
# =====================================

def read_answers_from_image_bgr(
    img_bgr,
    all_slots,
    slot_mapping,
    conf_thres: float = CONF_THRES,
    draw_template_points: bool = True
):
    results = model.predict(source=img_bgr, conf=conf_thres, verbose=False)
    det = results[0]

    boxes = det.boxes.xyxy.cpu().numpy()
    confs = det.boxes.conf.cpu().numpy()
    print(f"[60Q] YOLO marks: {len(boxes)}")

    answers = {q: None for q in range(1, NUM_QUESTIONS + 1)}
    marks_by_q = {q: {} for q in range(1, NUM_QUESTIONS + 1)}
    debug_img = img_bgr.copy()

    # debug: วาด template ทุกจุด (เปิด/ปิดได้)
    if draw_template_points:
        for _, (sx, sy) in all_slots.items():
            cv2.circle(debug_img, (int(sx), int(sy)), 3, (255, 100, 0), -1)

    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        conf = float(conf)
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0

        slot_idx = find_nearest_slot(xc, yc, all_slots, max_dist=MAX_SLOT_DIST)
        if slot_idx is None:
            continue

        qopt = slot_mapping.get(slot_idx)
        if qopt is None:
            continue

        qnum, opt = qopt

        prev_conf = marks_by_q[qnum].get(opt, 0.0)
        if conf > prev_conf:
            marks_by_q[qnum][opt] = conf

        cv2.putText(
            debug_img,
            f"{qnum}{opt} {conf:.2f}",
            (int(xc), int(yc)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    for q in range(1, NUM_QUESTIONS + 1):
        opts_conf = marks_by_q[q]
        if not opts_conf:
            continue
        if len(opts_conf) == 1:
            answers[q] = max(opts_conf.items(), key=lambda x: x[1])[0]
        else:
            answers[q] = "MULTI"

    return answers, debug_img

# =====================================
# GRADING
# =====================================

def grade_answers(answers: dict, answer_key: dict):
    correct = 0
    total = 0
    detail = {}
    blank = 0
    multi = 0
    wrong = 0

    for q in range(1, NUM_QUESTIONS + 1):
        correct_opt = answer_key.get(q)
        if correct_opt is None:
            continue

        total += 1
        stu_ans = answers.get(q)

        if stu_ans is None:
            detail[q] = ("-", None, correct_opt)
            blank += 1
        elif stu_ans == "MULTI":
            detail[q] = ("M", stu_ans, correct_opt)
            multi += 1
        elif stu_ans == correct_opt:
            detail[q] = ("✔", stu_ans, correct_opt)
            correct += 1
        else:
            detail[q] = ("✘", stu_ans, correct_opt)
            wrong += 1

    stats = {"correct": correct, "wrong": wrong, "blank": blank, "multi": multi, "total": total}
    return correct, total, detail, stats

def parse_answer_key_string(s: str):
    s_clean = "".join(ch.upper() for ch in s if ch.upper() in OPTIONS)
    key = {}
    for i, ch in enumerate(s_clean, start=1):
        if i > NUM_QUESTIONS:
            break
        key[i] = ch
    return key

# =====================================
# MAIN ENTRY สำหรับ app.py
# =====================================

def process_auto(img_bgr, answer_key_str: str):
    all_slots, slot_mapping = get_template_and_mapping()

    answers, debug_img = read_answers_from_image_bgr(
        img_bgr, all_slots, slot_mapping
    )

    effective_key = parse_answer_key_string(answer_key_str) if answer_key_str else ANSWER_KEY_DEFAULT

    if effective_key:
        _, _, detail, stats = grade_answers(answers, effective_key)
    else:
        blank = sum(1 for v in answers.values() if v is None)
        multi = sum(1 for v in answers.values() if v == "MULTI")
        answered = NUM_QUESTIONS - blank
        stats = {"correct": 0, "wrong": 0, "blank": blank, "multi": multi, "total": answered}
        detail = {}

    return answers, effective_key, detail, stats, debug_img

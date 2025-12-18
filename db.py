# db.py
import os
import sqlite3
from datetime import datetime

# =========================
# DB PATH (Render-friendly)
# =========================
# ตั้งใน Render → Environment: DB_PATH=/var/data/scangrade.db
DB_PATH = os.environ.get("DB_PATH", "scangrade.db")


# =========================
# PACKAGES
# =========================
PACKAGES = {
    "300 ครั้ง": {"credits": 300, "price": 69},
    "600 ครั้ง": {"credits": 600, "price": 99},
    "1000 ครั้ง": {"credits": 1000, "price": 199},
    "3000 ครั้ง": {"credits": 3000, "price": 299},
}


# =========================
# DB CONNECTION
# =========================
def get_db_connection():
    # สร้างโฟลเดอร์อัตโนมัติ (กัน path ไม่อยู่)
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass

    return conn


# =========================
# INIT DB
# =========================
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # USERS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            credits INTEGER NOT NULL DEFAULT 0,
            used_free INTEGER NOT NULL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    # ORDERS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            package TEXT NOT NULL,
            credits INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            slip_filename TEXT,
            slip_ref TEXT,
            status TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_slip_ref_unique
        ON orders(slip_ref)
    """)

    # DEVICES (Free trial)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            used_free INTEGER NOT NULL DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    # SAVED ANSWER KEYS
    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            subject TEXT NOT NULL,
            num_questions INTEGER NOT NULL,
            key_str TEXT NOT NULL,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)

    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_keys_unique
        ON saved_keys(username, subject, num_questions)
    """)

    conn.commit()
    conn.close()


# =========================
# USERS
# =========================
def get_user(username):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(username, initial_credits):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO users (username, credits, used_free, created_at, updated_at)
        VALUES (?, ?, 0, ?, ?)
    """, (username, int(initial_credits), now, now))
    conn.commit()
    conn.close()


def set_user_credits(username, new_credits):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    conn.execute("""
        UPDATE users SET credits = ?, updated_at = ?
        WHERE username = ?
    """, (int(new_credits), now, username))
    conn.commit()
    conn.close()


def adjust_user_credits(username, delta):
    """เพิ่ม/ลดเครดิต (ไม่ให้ติดลบ)"""
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT credits FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if not row:
            conn.execute("ROLLBACK")
            return None

        current = int(row["credits"])
        new_credits = current + int(delta)
        if new_credits < 0:
            new_credits = 0

        conn.execute("""
            UPDATE users SET credits = ?, updated_at = ?
            WHERE username = ?
        """, (new_credits, now, username))

        conn.execute("COMMIT")
        return new_credits
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def list_users(q="", sort="updated_at", direction="desc", limit=1000, offset=0):
    q = (q or "").strip().lower()

    allowed_sort = {"username", "credits", "created_at", "updated_at", "used_free"}
    sort = sort if sort in allowed_sort else "updated_at"
    direction = "asc" if (direction or "").lower() == "asc" else "desc"

    where = ""
    params = []

    if q:
        where = "WHERE lower(username) LIKE ?"
        params.append(f"%{q}%")

    sql = f"""
        SELECT username, credits, used_free, created_at, updated_at
        FROM users
        {where}
        ORDER BY {sort} {direction}
        LIMIT ? OFFSET ?
    """
    params.extend([int(limit), int(offset)])

    conn = get_db_connection()
    rows = conn.execute(sql, tuple(params)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =========================
# DEVICES (Free trial)
# =========================
def get_device(device_id):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM devices WHERE device_id = ?",
        (device_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_device(device_id):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO devices (device_id, used_free, created_at, updated_at)
        VALUES (?, 0, ?, ?)
        ON CONFLICT(device_id)
        DO UPDATE SET updated_at = excluded.updated_at
    """, (device_id, now, now))
    conn.commit()
    conn.close()


def mark_device_used_free(device_id):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    conn.execute("""
        UPDATE devices SET used_free = 1, updated_at = ?
        WHERE device_id = ?
    """, (now, device_id))
    conn.commit()
    conn.close()


# =========================
# ORDERS
# =========================
def is_slip_ref_used(slip_ref):
    if not slip_ref:
        return False
    conn = get_db_connection()
    row = conn.execute(
        "SELECT 1 FROM orders WHERE slip_ref = ? LIMIT 1",
        (slip_ref,)
    ).fetchone()
    conn.close()
    return row is not None


def create_order(username, pkg_key, slip_filename, slip_ref):
    if pkg_key not in PACKAGES:
        raise ValueError("INVALID_PACKAGE")

    if slip_ref and is_slip_ref_used(slip_ref):
        raise ValueError("SLIP_ALREADY_USED")

    pkg = PACKAGES[pkg_key]
    now = datetime.utcnow().isoformat()

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO orders
            (username, package, credits, amount, slip_filename, slip_ref, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        """, (
            username,
            pkg_key,
            pkg["credits"],
            pkg["price"],
            slip_filename,
            slip_ref,
            now,
            now
        ))
        order_id = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("SLIP_ALREADY_USED")
    finally:
        conn.close()

    return order_id


def approve_order_and_add_credits(order_id):
    conn = get_db_connection()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN")

        order = conn.execute(
            "SELECT * FROM orders WHERE id = ?",
            (order_id,)
        ).fetchone()
        if not order or order["status"] == "approved":
            conn.execute("ROLLBACK")
            return None, None

        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (order["username"],)
        ).fetchone()
        if not user:
            conn.execute("ROLLBACK")
            return None, None

        new_credits = user["credits"] + order["credits"]
        now = datetime.utcnow().isoformat()

        conn.execute("""
            UPDATE users SET credits = ?, updated_at = ?
            WHERE username = ?
        """, (new_credits, now, user["username"]))

        conn.execute("""
            UPDATE orders SET status = 'approved', updated_at = ?
            WHERE id = ?
        """, (now, order_id))

        conn.execute("COMMIT")
        return dict(order), {
            "username": user["username"],
            "credits": new_credits
        }
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# =========================
# SAVED ANSWER KEYS
# =========================
def upsert_saved_key(username, subject, num_questions, key_str):
    now = datetime.utcnow().isoformat()
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO saved_keys
        (username, subject, num_questions, key_str, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(username, subject, num_questions)
        DO UPDATE SET key_str = excluded.key_str, updated_at = excluded.updated_at
    """, (username, subject, num_questions, key_str, now, now))
    conn.commit()
    conn.close()


def get_saved_key(username, subject, num_questions):
    conn = get_db_connection()
    row = conn.execute("""
        SELECT key_str FROM saved_keys
        WHERE username = ? AND subject = ? AND num_questions = ?
    """, (username, subject, num_questions)).fetchone()
    conn.close()
    return row["key_str"] if row else None


def list_saved_subjects(username, num_questions):
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT subject, updated_at
        FROM saved_keys
        WHERE username = ? AND num_questions = ?
        ORDER BY updated_at DESC
    """, (username, num_questions)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


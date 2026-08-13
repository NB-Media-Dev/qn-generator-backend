import threading
import re
import difflib
import argparse
from collections import defaultdict

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)

DATABASE_URL = "sqlite:///./questions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30})

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class QuestionRecord(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)

    dy_ques_id = Column(String, nullable=True, index=True)
    dy_code = Column(String, nullable=True)
    ln_code = Column(String, nullable=True)
    dy_order = Column(String, nullable=True)
    dy_pattern = Column(String, nullable=True)
    dy_seconds = Column(String, nullable=True)
    dy_question = Column(String, nullable=False)
    dy_image_name = Column(String, nullable=True)
    dy_ans_1 = Column(String, nullable=False)
    dy_ans_2 = Column(String, nullable=False)
    dy_ans_3 = Column(String, nullable=False)
    dy_ans_4 = Column(String, nullable=False)
    dy_correct_ans = Column(String, nullable=False)                          
    dy_explain = Column(String, nullable=True)
    dy_explain_image_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utc_now_naive)
    updated_at = Column(DateTime, nullable=True)

    difficulty = Column(String, nullable=False)
    source_filename = Column(String, nullable=True)
    batch_id = Column(String, nullable=True, index=True)
    model = Column(String, nullable=True)
    board = Column(String, nullable=True, index=True)
    standard = Column(String, nullable=True, index=True)
    group_name = Column(String, nullable=True, index=True)
    subject = Column(String, nullable=True, index=True)

class SelectedQuestion(Base):
    __tablename__ = "selected_questions"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, unique=True, index=True)
    batch_id = Column(String, nullable=True, index=True)
    selected_at = Column(DateTime, default=_utc_now_naive)

class SubjectQuesCounter(Base):
    __tablename__ = "subject_ques_counters"

    prefix = Column(String, primary_key=True)                         
    last_number = Column(Integer, nullable=False, default=0)

class ExamOrderCounter(Base):
    __tablename__ = "exam_order_counters"

    dy_code = Column(String, primary_key=True)
    last_order = Column(Integer, nullable=False, default=0)

class LivePushLog(Base):
    __tablename__ = "live_push_log"

    dy_ques_id = Column(String, primary_key=True)
    exam_type = Column(String, primary_key=True, default="online")
    live_ques_id = Column(String, nullable=True)
    status = Column(String, nullable=False)
    response_detail = Column(String, nullable=True)
    pushed_at = Column(DateTime, default=_utc_now_naive)

Base.metadata.create_all(bind=engine)

with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.execute(text("PRAGMA synchronous=NORMAL"))
    conn.commit()

    existing_columns = [row[1] for row in conn.execute(text("PRAGMA table_info(questions)"))]

    _renames = [
        ("question_text", "dy_question"),
        ("option_a", "dy_ans_1"),
        ("option_b", "dy_ans_2"),
        ("option_c", "dy_ans_3"),
        ("option_d", "dy_ans_4"),
    ]
    for old_name, new_name in _renames:
        if old_name in existing_columns and new_name not in existing_columns:
            conn.execute(text(f"ALTER TABLE questions RENAME COLUMN {old_name} TO {new_name}"))
            conn.commit()
            existing_columns.append(new_name)

    _new_columns = [
        ("dy_ques_id", "VARCHAR"),
        ("dy_code", "VARCHAR"),
        ("ln_code", "VARCHAR"),
        ("dy_order", "VARCHAR"),
        ("dy_pattern", "VARCHAR"),
        ("dy_seconds", "VARCHAR"),
        ("dy_image_name", "VARCHAR"),
        ("dy_correct_ans", "VARCHAR"),
        ("dy_explain", "VARCHAR"),
        ("dy_explain_image_name", "VARCHAR"),
        ("updated_at", "DATETIME"),
        ("difficulty", "VARCHAR"),
        ("source_filename", "VARCHAR"),
        ("batch_id", "VARCHAR"),
        ("model", "VARCHAR"),
        ("board", "VARCHAR"),
        ("standard", "VARCHAR"),
        ("group_name", "VARCHAR"),
        ("subject", "VARCHAR"),
    ]
    for col_name, col_type in _new_columns:
        if col_name not in existing_columns:
            conn.execute(text(f"ALTER TABLE questions ADD COLUMN {col_name} {col_type}"))
            conn.commit()

    if "correct_answer" in existing_columns:
        conn.execute(text("""
            UPDATE questions
            SET dy_correct_ans = CASE correct_answer
                WHEN 'A' THEN '1' WHEN 'B' THEN '2'
                WHEN 'C' THEN '3' WHEN 'D' THEN '4'
                ELSE dy_correct_ans END
            WHERE dy_correct_ans IS NULL OR dy_correct_ans = ''
        """))
        conn.commit()

    existing_selected_columns = [row[1] for row in conn.execute(text("PRAGMA table_info(selected_questions)"))]
    if "batch_id" not in existing_selected_columns:
        conn.execute(text("ALTER TABLE selected_questions ADD COLUMN batch_id VARCHAR"))
        conn.commit()

_counter_lock = threading.Lock()

def get_next_ques_number(prefix: str) -> int:
    with _counter_lock:
        db = SessionLocal()
        try:
            row = db.query(SubjectQuesCounter).filter(SubjectQuesCounter.prefix == prefix).first()
            if row is None:
                row = SubjectQuesCounter(prefix=prefix, last_number=0)
                db.add(row)
            row.last_number += 1
            next_number = row.last_number
            db.commit()
            return next_number
        finally:
            db.close()

def get_next_ques_numbers_batch(prefix: str, count: int, db=None) -> list[int]:
    with _counter_lock:
        local_session = False
        if db is None:
            db = SessionLocal()
            local_session = True
        try:
            row = db.query(SubjectQuesCounter).filter(SubjectQuesCounter.prefix == prefix).first()
            if row is None:
                row = SubjectQuesCounter(prefix=prefix, last_number=0)
                db.add(row)
            start_num = row.last_number + 1
            row.last_number += count
            if local_session:
                db.commit()
            return list(range(start_num, start_num + count))
        finally:
            if local_session:
                db.close()

def get_next_order_numbers_batch(dy_code: str, count: int, db=None) -> list[int]:
    with _counter_lock:
        local_session = False
        if db is None:
            db = SessionLocal()
            local_session = True
        try:
            # Query actual max order present in DB for this dy_code
            existing_orders = db.query(QuestionRecord.dy_order).filter(QuestionRecord.dy_code == dy_code).all()
            actual_max = 0
            for (ord_val,) in existing_orders:
                if ord_val and str(ord_val).isdigit():
                    actual_max = max(actual_max, int(ord_val))

            row = db.query(ExamOrderCounter).filter(ExamOrderCounter.dy_code == dy_code).first()
            if row is None:
                row = ExamOrderCounter(dy_code=dy_code, last_order=actual_max)
                db.add(row)
            else:
                # Sync counter down if questions were deleted
                if row.last_order > actual_max:
                    row.last_order = actual_max

            start_num = row.last_order + 1
            row.last_order += count
            if local_session:
                db.commit()
            return list(range(start_num, start_num + count))
        finally:
            if local_session:
                db.close()

ALLOWED_EXTRA_CHARS = (
    "\u2018\u2019\u201C\u201D\u2013\u2014\u2026\u00A0\u200c\u200d"
    "\u00d7\u00f7\u2212\u00b1\u00b0\u2032\u2033\u221a\u03c0"
    "\u00bc\u00bd\u00be\u00b2\u00b3\u20b9"
)
LANGUAGE_SCRIPT_RANGES = {"Tamil": [(0x0B80, 0x0BFF)]}
TAMIL_CHAR_RE = re.compile("[\u0B80-\u0BFF]")
LATIN_CHAR_RE = re.compile("[A-Za-z]")
NEAR_DUPLICATE_THRESHOLD = 0.8

def detect_dominant_language(text_value: str) -> str:
    tamil_count = len(TAMIL_CHAR_RE.findall(text_value))
    latin_count = len(LATIN_CHAR_RE.findall(text_value))
    total_alpha = tamil_count + latin_count
    if total_alpha == 0:
        return 'English'
    return 'Tamil' if tamil_count / total_alpha > 0.5 else 'English'

def has_wrong_script_chars(text_value: str, language: str) -> bool:
    if not text_value:
        return False
    ranges = LANGUAGE_SCRIPT_RANGES.get(language, [])
    for ch in text_value:
        if ch in ALLOWED_EXTRA_CHARS:
            continue
        code = ord(ch)
        if code < 0x80:
            continue
        if any(low <= code <= high for low, high in ranges):
            continue
        return True
    return False

def record_has_script_corruption(record: "QuestionRecord") -> bool:
    fields = [record.dy_question, record.dy_explain, record.dy_ans_1, record.dy_ans_2, record.dy_ans_3, record.dy_ans_4]
    present_fields = [field for field in fields if field]
    language = detect_dominant_language(" ".join(present_fields))
    if language not in LANGUAGE_SCRIPT_RANGES:
        return False
    return any(has_wrong_script_chars(field, language) for field in present_fields)

def normalize_question_key(text_value: str) -> str:
    return re.sub(r"\s+", " ", (text_value or "").strip().lower())

def find_duplicate_ids(records: list) -> set:
    ordered_records = sorted(records, key=lambda record: record.id)
    kept_keys = []
    duplicate_ids = set()
    for record in ordered_records:
        key = normalize_question_key(record.dy_question)
        if not key:
            continue
        is_duplicate = any(
            difflib.SequenceMatcher(None, key, kept_key).ratio() >= NEAR_DUPLICATE_THRESHOLD
            for _, kept_key in kept_keys
        )
        if is_duplicate:
            duplicate_ids.add(record.id)
        else:
            kept_keys.append((record.id, key))
    return duplicate_ids

def find_bad_question_ids(source_filename: str = None):
    db = SessionLocal()
    try:
        query = db.query(QuestionRecord)
        if source_filename:
            query = query.filter(QuestionRecord.source_filename == source_filename)
        all_records = query.order_by(QuestionRecord.id).all()

        groups = defaultdict(list)
        for record in all_records:
            groups[(record.source_filename, record.difficulty, record.standard, record.subject)].append(record)

        corrupted_ids = set()
        duplicate_ids = set()
        for group_key, group_records in groups.items():
            group_corrupted_ids = {record.id for record in group_records if record_has_script_corruption(record)}
            corrupted_ids |= group_corrupted_ids
            clean_records = [record for record in group_records if record.id not in group_corrupted_ids]
            group_duplicate_ids = find_duplicate_ids(clean_records)
            duplicate_ids |= group_duplicate_ids
            flagged_count = len(group_corrupted_ids) + len(group_duplicate_ids)
            if flagged_count:
                source, difficulty, standard, subject = group_key
                print(f"[{source or '(no file)'}] {standard} / {subject} / {difficulty}: "
                      f"{len(group_records)} rows -> {len(group_corrupted_ids)} script-corrupted, "
                      f"{len(group_duplicate_ids)} duplicates")

        return len(all_records), corrupted_ids, duplicate_ids
    finally:
        db.close()

def delete_question_ids(question_ids: set):
    db = SessionLocal()
    try:
        db.query(SelectedQuestion).filter(SelectedQuestion.question_id.in_(question_ids)).delete(synchronize_session=False)
        deleted_count = db.query(QuestionRecord).filter(QuestionRecord.id.in_(question_ids)).delete(synchronize_session=False)
        db.commit()
        return deleted_count
    finally:
        db.close()

def clean_up_questions(apply_changes: bool = False, source_filename: str = None):
    total_rows, corrupted_ids, duplicate_ids = find_bad_question_ids(source_filename)
    flagged_ids = corrupted_ids | duplicate_ids

    print()
    print(f"Total rows scanned: {total_rows}")
    print(f"Script-corrupted:   {len(corrupted_ids)}")
    print(f"Duplicates:         {len(duplicate_ids)}")
    print(f"Flagged overall:    {len(flagged_ids)}")

    if not flagged_ids:
        print("Nothing to clean up.")
        return

    if not apply_changes:
        print("\nDry run only - no rows deleted. Re-run with --apply to delete the flagged rows.")
        return

    deleted_count = delete_question_ids(flagged_ids)
    print(f"\nDeleted {deleted_count} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--source-filename", default=None)
    args = parser.parse_args()
    clean_up_questions(apply_changes=args.apply, source_filename=args.source_filename)
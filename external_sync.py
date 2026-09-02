import os
import json
import re
import asyncio
from datetime import datetime, timezone
from typing import Iterable, List, Literal, Optional

import httpx

from database import SessionLocal, QuestionRecord, LivePushLog

EXTERNAL_DB_API_URL = os.getenv('EXTERNAL_DB_API_URL') or os.getenv('LIVE_DB_API_URL', '')
EXTERNAL_API_KEY = os.getenv('EXTERNAL_API_KEY') or os.getenv('LIVE_DB_API_TOKEN', '')
LIVE_PUSH_DEBUG = os.getenv('LIVE_PUSH_DEBUG', '0') == '1'

EXAM_TYPE_MAP = {
    'daily': 'daily',
    'schedule': 'schedule',
    'online': 'online',
}

MAX_CONCURRENCY = 1  

LETTER_TO_INT = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4,
    '1': 1, '2': 2, '3': 3, '4': 4
}

_SUCCESS_KEYS = ('success', 'ok', 'issuccess', 'inserted', 'status')
_FAILURE_KEYS = ('error', 'errors', 'errormessage', 'message')
_SUCCESS_STATUS_VALUES = {'success', 'ok', 'created', 'inserted', 'true', '1'}
_FAILURE_STATUS_VALUES = {'failed', 'failure', 'error', 'false', '0', 'duplicate', 'rejected'}


def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _response_indicates_success(http_ok: bool, detail) -> tuple[bool, str]:
    if not http_ok:
        return False, 'non-2xx HTTP status'

    if not isinstance(detail, dict):
        return True, 'no JSON body to verify; trusted HTTP status only'

    lower_map = {str(k).lower(): v for k, v in detail.items()}

    for key in _SUCCESS_KEYS:
        if key in lower_map:
            val = lower_map[key]
            if isinstance(val, bool):
                return val, f'body.{key}={val}'
            if isinstance(val, str):
                v = val.strip().lower()
                if v in _SUCCESS_STATUS_VALUES:
                    return True, f'body.{key}={v!r}'
                if v in _FAILURE_STATUS_VALUES:
                    return False, f'body.{key}={v!r}'

    for key in _FAILURE_KEYS:
        if key in lower_map and lower_map[key]:
            return False, f'body.{key}={lower_map[key]!r}'

    return True, 'body present but no recognizable success/error field; trusted HTTP status only'


_NUMERIC_RE = re.compile(r'-?\d+(?:\.\d+)?')
DEFAULT_QS_SECONDS = 60  


def _to_numeric_seconds(raw, dy_ques_id: str = '') -> int:
    if raw is not None:
        match = _NUMERIC_RE.search(str(raw))
        if match:
            num_str = match.group(0)
            return int(round(float(num_str)))

    print(f"[live-push] WARNING quesId={dy_ques_id!r}: dy_seconds={raw!r} has no numeric value, "
          f"defaulting qsSeconds to {DEFAULT_QS_SECONDS}. Fix the source data if this is unexpected.")
    return DEFAULT_QS_SECONDS


def row_to_live_payload(record: QuestionRecord, exam_type: str, exam_code: Optional[str] = None) -> dict:
    raw_ans = str(record.dy_correct_ans).strip().upper()
    crt_ans = LETTER_TO_INT.get(raw_ans, 1)

    return {
        'examType': EXAM_TYPE_MAP[exam_type],
        'examCode': exam_code or record.dy_code,
        'quesId': record.dy_ques_id,
        'lnCode': record.ln_code,
        'qsOrder': int(record.dy_order) if str(record.dy_order).isdigit() else record.dy_order,
        'qsPattern': int(record.dy_pattern) if record.dy_pattern is not None and str(record.dy_pattern).isdigit() else (record.dy_pattern or 1),
        'qsSeconds': _to_numeric_seconds(record.dy_seconds, record.dy_ques_id),
        'qsQuestion': record.dy_question,
        'option1': record.dy_ans_1,
        'option2': record.dy_ans_2,
        'option3': record.dy_ans_3,
        'option4': record.dy_ans_4,
        'crtAns': crt_ans,
        'qsExplain': record.dy_explain,
    }


def _payload_missing_fields(payload: dict) -> list:
    required = ['examType', 'examCode', 'quesId', 'qsQuestion', 'option1', 'option2', 'option3', 'option4', 'crtAns']
    missing = [key for key in required if payload.get(key) in (None, '')]
    return missing


async def post_question_to_live_db(client: httpx.AsyncClient, payload: dict):
    try:
        resp = await client.post(
            EXTERNAL_DB_API_URL,
            json=payload,
            headers={
                'Authorization': f'Bearer {EXTERNAL_API_KEY}',
                'Content-Type': 'application/json',
            },
            timeout=20.0,
        )
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text

        http_ok = 200 <= resp.status_code < 300
        ok, reason = _response_indicates_success(http_ok, detail)

        if LIVE_PUSH_DEBUG or not ok:
            print(
                f"[live-push] quesId={payload.get('quesId')!r} "
                f"http_status={resp.status_code} ok={ok} reason={reason!r}\n"
                f"[live-push] raw response: {detail!r}"
            )

        if not ok:
            detail = {'_verdict': 'rejected', '_reason': reason, 'raw_response': detail}

        return ok, detail
    except httpx.RequestError as e:
        return False, f'Network error contacting live DB: {str(e)}'


def get_already_pushed(dy_ques_ids: List[str], exam_type: str) -> set:
    if not dy_ques_ids:
        return set()
    db = SessionLocal()
    try:
        rows = (
            db.query(LivePushLog.dy_ques_id)
            .filter(
                LivePushLog.exam_type == exam_type,
                LivePushLog.status == 'success',
                LivePushLog.dy_ques_id.in_(dy_ques_ids),
            )
            .all()
        )
        return {r[0] for r in rows}
    finally:
        db.close()


def log_push_results_batch(results_list: List[dict], exam_type: str) -> None:
    if not results_list:
        return
    db = SessionLocal()
    try:
        now = _utc_now_naive()
        dy_ids = [item['dy_ques_id'] for item in results_list]

        existing_logs = {
            log.dy_ques_id: log
            for log in db.query(LivePushLog)
            .filter(LivePushLog.exam_type == exam_type, LivePushLog.dy_ques_id.in_(dy_ids))
            .all()
        }

        for item in results_list:
            dy_ques_id = item['dy_ques_id']
            live_ques_id = item['live_ques_id']
            ok = item['ok']
            detail = item['detail']
            detail_str = json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail

            if dy_ques_id in existing_logs:
                existing = existing_logs[dy_ques_id]
                existing.live_ques_id = live_ques_id
                existing.status = 'success' if ok else 'failed'
                existing.response_detail = detail_str
                existing.pushed_at = now
            else:
                db.add(LivePushLog(
                    dy_ques_id=dy_ques_id,
                    exam_type=exam_type,
                    live_ques_id=live_ques_id,
                    status='success' if ok else 'failed',
                    response_detail=detail_str,
                    pushed_at=now,
                ))
        db.commit()
    finally:
        db.close()


async def push_records_to_live_db(
    records: Iterable[QuestionRecord],
    exam_type: Literal['daily', 'schedule', 'online'],
    exam_code: Optional[str] = None,
    force: bool = False,
) -> dict:
    if not EXTERNAL_DB_API_URL or not EXTERNAL_API_KEY:
        raise RuntimeError('EXTERNAL_DB_API_URL / EXTERNAL_API_KEY are not configured on the server (.env).')

    records = list(records)
    try:
        records = sorted(records, key=lambda r: int(r.dy_order) if r.dy_order and str(r.dy_order).isdigit() else 0)
    except Exception:
        pass  
    
    all_ids = [r.dy_ques_id for r in records]
    already_pushed = set() if force else get_already_pushed(all_ids, exam_type)
    to_push = [r for r in records if r.dy_ques_id not in already_pushed]

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _push_single(client: httpx.AsyncClient, record: QuestionRecord):
        async with semaphore:
            payload = row_to_live_payload(record, exam_type, exam_code)

            missing = _payload_missing_fields(payload)
            if missing:
                print(f"[live-push] SKIPPING quesId={payload.get('quesId')!r}: missing fields {missing}")
                return {
                    'record': record,
                    'dy_ques_id': record.dy_ques_id,
                    'live_ques_id': payload.get('quesId'),
                    'ok': False,
                    'detail': {'_verdict': 'not_sent', '_reason': f'missing required fields: {missing}'},
                }

            ok, detail = await post_question_to_live_db(client, payload)
            return {
                'record': record,
                'dy_ques_id': record.dy_ques_id,
                'live_ques_id': payload['quesId'],
                'ok': ok,
                'detail': detail,
            }

    async with httpx.AsyncClient() as client:
        tasks = [_push_single(client, record) for record in to_push]
        raw_results = await asyncio.gather(*tasks)

    log_push_results_batch(raw_results, exam_type)

    pushed, failed = [], []
    for res in raw_results:
        entry = {
            'dy_ques_id': res['dy_ques_id'],
            'quesId': res['live_ques_id'],
            'response': res['detail'],
        }
        (pushed if res['ok'] else failed).append(entry)

    return {
        'status': 'success',
        'exam_type': EXAM_TYPE_MAP[exam_type],
        'total_selected': len(records),
        'skipped_already_pushed': len(already_pushed),
        'pushed_count': len(pushed),
        'failed_count': len(failed),
        'pushed': pushed,
        'failed': failed,
    }

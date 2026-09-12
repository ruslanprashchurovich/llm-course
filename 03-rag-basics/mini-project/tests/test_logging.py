"""Структурные логи (урок 3.7): request_id сшивает события, вопрос - только хэшем.

Приём: conftest настраивает structlog на io.StringIO, тест читает
JSONL-строки как данные - логи проверяются так же строго, как ответы API.
"""

import json

from app.logs import question_fingerprint


def read_events(stream) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_ask_writes_request_received_and_answer_returned(client, log_stream):
    client.post("/ask", json={"question": "Сколько дней отпуска положено?"})
    events = {e["event"]: e for e in read_events(log_stream)}
    assert "request_received" in events
    assert "answer_returned" in events
    assert events["answer_returned"]["verdict"] == "ok"
    assert events["answer_returned"]["llm_calls"] == 2


def test_request_id_stitches_all_events_of_one_request(client, log_stream):
    client.post("/ask", json={"question": "Сколько дней отпуска положено?"})
    events = read_events(log_stream)
    assert len(events) >= 3            # received, retrieval_done, answer_returned
    request_ids = {e.get("request_id") for e in events}
    assert len(request_ids) == 1 and None not in request_ids


def test_each_request_gets_its_own_request_id(client, log_stream):
    client.post("/ask", json={"question": "Сколько дней отпуска положено?"})
    client.post("/ask", json={"question": "Когда разрешён деплой в production?"})
    request_ids = {e["request_id"] for e in read_events(log_stream)}
    assert len(request_ids) == 2


def test_question_reaches_log_only_as_hash(client, log_stream):
    secret = "Сколько платят за сверхурочные Драконоборцеву?"
    client.post("/ask", json={"question": secret})
    raw_log = log_stream.getvalue()
    assert "Драконоборцеву" not in raw_log            # текста вопроса нет нигде
    received = next(e for e in read_events(log_stream)
                    if e["event"] == "request_received")
    assert received["question_hash"] == question_fingerprint(secret)["question_hash"]
    assert received["question_len"] == len(secret)


def test_fingerprint_ignores_case_and_outer_spaces():
    a = question_fingerprint("Сколько дней отпуска?")
    b = question_fingerprint("  сколько дней отпуска?  ")
    assert a["question_hash"] == b["question_hash"]   # повторы группируются
    assert a["question_hash"] != question_fingerprint("Другой вопрос?")["question_hash"]

"""Задание 3: судья-код триажа, конвейеры команды/одиночки и сводка eval'а — без сети."""

import json

import pytest

from team.llm import ScriptedLLM
from team.log import TeamLog
from team.triage import (
    TICKETS,
    TRIAGE_ROLE_OF_SYSTEM,
    EvalSummary,
    Triage,
    TriageJudge,
    compare_table,
    evaluate,
    is_grounded,
    normalize,
    paid_off,
    parse_facts,
    parse_triage,
    policy_violations,
    run_solo,
    run_team,
)

judge = TriageJudge()


def gold_answer(ticket, **overrides) -> str:
    data = {"category": ticket.category, "priority": ticket.priority, "facts": [ticket.gold_fact],
            "needs_human": ticket.needs_human, "summary": "Короткое резюме тикета своими словами."}
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


# ------------------------------------------------------------------ eval-набор и судья
@pytest.mark.parametrize("ticket", TICKETS, ids=[t.id for t in TICKETS])
def test_gold_labels_are_consistent_with_the_rules_judge(ticket):
    """Золотая разметка обязана проходить собственного судью — иначе судья спорит с eval'ом."""
    verdict = judge(ticket.text, gold_answer(ticket))
    assert verdict.ok, verdict.report


def test_eval_set_covers_every_category_and_priority():
    assert {t.category for t in TICKETS} == {"outage", "billing", "access", "performance",
                                             "question", "other"}
    assert {t.priority for t in TICKETS} == {"low", "normal", "high", "critical"}
    assert {t.needs_human for t in TICKETS} == {True, False}
    assert len({t.id for t in TICKETS}) == len(TICKETS) == 12


def test_normalize_ignores_case_yo_quotes_dashes_and_spaces():
    assert normalize("Выводится  «АККАУНТ ЗАБЛОКИРОВАН».") == "выводится аккаунт заблокирован"
    assert normalize("8–10 секунд") == normalize("8-10 секунд")
    assert is_grounded('"всё сломалось"', "ВСЁ СЛОМАЛОСЬ!!!")
    assert not is_grounded("сайт лежит", "сайт грузится долго")
    assert not is_grounded("!!!", "ВСЁ СЛОМАЛОСЬ!!!")          # слишком коротко для факта


def test_parse_triage_accepts_fenced_json_and_explains_failures():
    fenced = "Вот ответ:\n```json\n" + gold_answer(TICKETS[0]) + "\n```"
    triage, detail = parse_triage(fenced)
    assert isinstance(triage, Triage) and detail == "схема соблюдена"

    assert parse_triage("нет никакого json")[0] is None
    _, detail = parse_triage('{"category": "outage", "priority": }')
    assert detail.startswith("JSON не парсится")
    _, detail = parse_triage(gold_answer(TICKETS[0], extra="поле"))
    assert "extra" in detail and "схема нарушена" in detail
    _, detail = parse_triage(gold_answer(TICKETS[0], priority="urgent"))
    assert "priority" in detail


def test_policy_rules_each_fire_on_their_own_violation():
    base = dict(facts=["получают ошибку 550 relay denied"], needs_human=True, summary="Резюме тикета.")
    r1 = policy_violations(Triage(category="outage", priority="low", **base), "почта не работает")
    r2 = policy_violations(Triage(category="billing", priority="critical", **base), "счёт")
    r3 = policy_violations(Triage(category="outage", priority="high", facts=["всё сломалось"],
                                  needs_human=True, summary="Резюме тикета."), "ВСЁ СЛОМАЛОСЬ")
    r4 = policy_violations(Triage(category="billing", priority="normal", facts=["списали дважды"],
                                  needs_human=False, summary="Резюме тикета."), "списали дважды")
    r5 = policy_violations(Triage(category="question", priority="high", facts=["ошибка 550"],
                                  needs_human=True, summary="Резюме тикета."), "как настроить")
    assert [v[:2] for v in r1] == ["R1"]
    assert [v[:2] for v in r2] == ["R2", "R7"]      # billing/critical нарушает оба
    assert [v[:2] for v in r3] == ["R3"]
    assert [v[:2] for v in r4] == ["R4"]
    assert [v[:2] for v in r5] == ["R5"]
    r6 = policy_violations(Triage(category="performance", priority="high", facts=["ошибка 500"],
                                  needs_human=False, summary="Резюме тикета."), "ошибка 500 раз в час")
    r7 = policy_violations(Triage(category="billing", priority="high", facts=["счёт № 48211"],
                                  needs_human=True, summary="Резюме тикета."), "счёт № 48211")
    assert [v[:2] for v in r6] == ["R6"]
    assert [v[:2] for v in r7] == ["R7"]


def test_judge_report_reads_as_pass_fail_lines():
    t = TICKETS[5]                                   # тикет-крик
    scream = gold_answer(t, category="outage", priority="critical", facts=["всё сломалось"])
    verdict = judge(t.text, scream)
    lines = verdict.report.splitlines()
    assert lines[0] == "[PASS] schema: схема соблюдена"
    assert lines[1].startswith("[PASS] grounded")
    assert lines[2].startswith("[FAIL] rules: R3") and not verdict.ok


def test_parse_facts_tolerates_prose_around_the_array():
    raw = 'Факты:\n["отдаёт 502 Bad Gateway", "теряем заказы", 42, ""]\nГотово.'
    assert parse_facts(raw) == ["отдаёт 502 Bad Gateway", "теряем заказы"]
    assert parse_facts("ничего") == []


# ------------------------------------------------------------------ конвейеры
def scripted(**replies) -> ScriptedLLM:
    return ScriptedLLM(replies=replies, role_of=TRIAGE_ROLE_OF_SYSTEM)


def test_solo_without_fix_only_scores_and_escalates():
    t = TICKETS[0]
    llm = scripted(solo=[gold_answer(t, facts=["выдуманный факт про базу"])])
    outcome = run_solo(t, llm, judge, fix_rounds=0, log=TeamLog(verbose=False))
    assert outcome.mode == "solo" and not outcome.accepted and outcome.escalated
    assert outcome.llm_calls == 1 and not outcome.verdict.passed("grounded")
    assert outcome.correct("category") and outcome.correct("priority")   # разметка верна, судья строг


def test_solo_with_fix_repairs_by_report():
    t = TICKETS[0]
    llm = scripted(solo=[gold_answer(t, facts=["выдуманный факт про базу"]), gold_answer(t)])
    outcome = run_solo(t, llm, judge, fix_rounds=1, log=TeamLog(verbose=False))
    assert outcome.mode == "solo+fix" and outcome.accepted and outcome.llm_calls == 2
    fix_call = llm.calls_by_role("solo")[1].user
    assert "[FAIL] grounded" in fix_call and "Твой предыдущий ответ" in fix_call


def test_team_drops_ungrounded_facts_before_the_triager():
    t = TICKETS[0]
    llm = scripted(extractor=['["отдаёт 502 Bad Gateway всем посетителям", "база данных упала"]'],
                   triager=[gold_answer(t)])
    outcome = run_team(t, llm, judge, log=TeamLog(verbose=False))
    assert outcome.accepted and outcome.facts_dropped == 1 and outcome.llm_calls == 2
    triager_call = llm.calls_by_role("triager")[0].user
    assert "отдаёт 502 Bad Gateway всем посетителям" in triager_call
    assert "база данных упала" not in triager_call


def test_team_escalates_to_human_after_fix_budget():
    t = TICKETS[5]
    bad = gold_answer(t, category="outage", priority="critical", facts=["всё сломалось"])
    llm = scripted(extractor=['["ВСЁ СЛОМАЛОСЬ"]'], triager=[bad, bad])
    log = TeamLog(verbose=False)
    outcome = run_team(t, llm, judge, fix_rounds=1, log=log)
    assert outcome.escalated and outcome.llm_calls == 3
    kinds = [k for _, _, k in log.shape()]
    assert kinds == ["facts", "json", "check_report", "json", "check_report"]


def test_evaluate_and_compare_table_and_verdict():
    tickets = TICKETS[:2]
    good = [gold_answer(t) for t in tickets]
    solo = evaluate("solo", tickets, scripted(solo=[gold_answer(tickets[0], facts=["выдумка тут"]),
                                                  gold_answer(tickets[1], facts=["выдумка там"])]),
                    judge, verbose=False)
    # ScriptedLLM повторяет последнюю реплику, поэтому одиночке даём один сценарий на тикет
    team = EvalSummary("team")
    for t, answer in zip(tickets, good):
        team.outcomes.append(run_team(t, scripted(extractor=[json.dumps([t.gold_fact], ensure_ascii=False)],
                                                  triager=[answer]), judge, log=TeamLog(verbose=False)))
    assert solo.grounded_ok == 0.0 and team.accepted == 1.0
    assert team.quality == 1.0 and team.llm_calls == 4
    table = compare_table([solo, team])
    assert table.splitlines()[0].split()[1:] == ["solo", "team"]
    assert "КАЧЕСТВО" in table
    assert "п.п." in paid_off(team, solo)

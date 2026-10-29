"""Relative-time notes: a turn stamped 2025-04-27 that says "yesterday" is rendered with "(dates: yesterday = 2025-04-26)".

Why: the weakest leaf on the whole Cycle-1 board was "Dates, Relative Time & Intervals" (18-21 % for every open-source
entry). The answer model has to combine the turn's timestamp with words like "last week" or "三周前"; writing the
resolved date next to the phrase makes that step explicit. Idea credited to InvMem (Cycle-1 #1,
github.com/wenxiaof345-ctrl/vanilla-rag-memory), which annotated English expressions on the stored text; the rules,
tables and the Chinese patterns here are our own, and we apply them at render time only (the index stays raw text).

Everything is derived from the turn's own timestamp, so no information is added that the conversation did not
contain. Granularity follows the phrase: days -> YYYY-MM-DD, weeks -> "week of <Monday>", months -> YYYY-MM,
years -> YYYY. Same-period phrases (today, this week, 今天, 本月 ...) are deliberately NOT annotated: the stamp in
front of the turn already carries that date, and the extra note was pure clutter in the 10-08 judge runs.
Nothing here reads the query, so it can never encode an answer.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

MAX_NOTES = 4
Resolver = Callable[[datetime], str]


def _day(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _week(d: datetime) -> str:
    return "week of " + _day(d - timedelta(days=d.weekday()))


def _month(d: datetime) -> str:
    return d.strftime("%Y-%m")


def _year(d: datetime) -> str:
    return d.strftime("%Y")


def _shift_months(d: datetime, n: int) -> datetime:
    total = d.year * 12 + (d.month - 1) + n
    year, month = divmod(total, 12)
    month += 1
    last = (datetime(year + (month // 12), (month % 12) + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)).day
    return d.replace(year=year, month=month, day=min(d.day, last))


def _shift_years(d: datetime, n: int) -> datetime:
    return _shift_months(d, 12 * n)


def _by_unit(unit: str, n: int) -> Resolver:
    """n may be negative (past). Days and weeks shift by days; months and years shift by calendar units."""
    if unit == "day":
        return lambda d: _day(d + timedelta(days=n))
    if unit == "week":
        return lambda d: _week(d + timedelta(days=7 * n))
    if unit == "month":
        return lambda d: _month(_shift_months(d, n))
    return lambda d: _year(_shift_years(d, n))


EN_NUMBER = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "couple of": 2}
EN_UNIT = {"day": "day", "week": "week", "month": "month", "year": "year"}
_EN_N = r"(?P<n>\d{1,3}|a couple of|couple of|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_EN_U = r"(?P<u>day|week|month|year)s?"

EN_FIXED: list[tuple[str, Resolver]] = [
    (r"the day before yesterday", _by_unit("day", -2)),
    (r"the day after tomorrow", _by_unit("day", 2)),
    (r"last night|yesterday", _by_unit("day", -1)),
    (r"tomorrow", _by_unit("day", 1)),
    (r"last week", _by_unit("week", -1)),
    (r"next week", _by_unit("week", 1)),
    (r"last month", _by_unit("month", -1)),
    (r"next month", _by_unit("month", 1)),
    (r"last year", _by_unit("year", -1)),
    (r"next year", _by_unit("year", 1)),
]
EN_AGO = re.compile(r"\b" + _EN_N + r"\s+" + _EN_U + r"\s+ago\b", re.IGNORECASE)
EN_AHEAD = re.compile(r"\b(?:in\s+" + _EN_N + r"\s+" + _EN_U + r"(?:'?\s*time)?|" + _EN_N.replace("?P<n>", "?P<n2>")
                      + r"\s+" + _EN_U.replace("?P<u>", "?P<u2>") + r"\s+from\s+now)\b", re.IGNORECASE)
EN_FIXED_RE = [(re.compile(r"\b(?:" + pat + r")\b", re.IGNORECASE), fn) for pat, fn in EN_FIXED]

CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"天": "day", "周": "week", "个星期": "week", "星期": "week", "个礼拜": "week", "礼拜": "week",
           "个月": "month", "年": "year"}
ZH_FIXED: list[tuple[str, Resolver]] = [
    (r"大前天", _by_unit("day", -3)),
    (r"前天", _by_unit("day", -2)),
    (r"昨天|昨晚|昨日", _by_unit("day", -1)),
    (r"明天|明日", _by_unit("day", 1)),
    (r"大后天", _by_unit("day", 3)),
    (r"后天", _by_unit("day", 2)),
    (r"上上周|上上个星期|上上星期", _by_unit("week", -2)),
    (r"上周|上星期|上个星期|上礼拜|上个礼拜", _by_unit("week", -1)),
    (r"下下周|下下个星期", _by_unit("week", 2)),
    (r"下周|下星期|下个星期|下礼拜|下个礼拜", _by_unit("week", 1)),
    (r"上上个月", _by_unit("month", -2)),
    (r"上个月|上月", _by_unit("month", -1)),
    (r"下下个月", _by_unit("month", 2)),
    (r"下个月|下月", _by_unit("month", 1)),
    (r"前年", _by_unit("year", -2)),
    (r"去年", _by_unit("year", -1)),
    (r"明年", _by_unit("year", 1)),
    (r"后年", _by_unit("year", 2)),
]
ZH_FIXED_RE = [(re.compile(pat), fn) for pat, fn in ZH_FIXED]
ZH_REL = re.compile(r"(?<!\d)(?P<n>\d{1,3}|[一二两三四五六七八九十]{1,3})\s*(?P<u>个星期|个礼拜|个月|星期|礼拜|天|周|年)\s*"
                    r"(?P<dir>以前|之前|前|以后|之后|后)")


def cn_number(text: str) -> Optional[int]:
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        tens, ones = text.split("十", 1)
        if (tens and tens not in CN_DIGIT) or (ones and ones not in CN_DIGIT):
            return None
        return (CN_DIGIT[tens] if tens else 1) * 10 + (CN_DIGIT[ones] if ones else 0)
    if len(text) == 1 and text in CN_DIGIT:
        return CN_DIGIT[text]
    return None


def relative_time_notes(text: str, event_ms: Optional[int]) -> list[str]:
    """Resolved relative-time phrases found in `text`, in order of appearance; [] without a timestamp."""
    if event_ms is None or not text:
        return []
    try:
        base = datetime.fromtimestamp(event_ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return []
    found: list[tuple[int, int, str, Resolver]] = []   # (start, end, phrase, resolver)

    def take(start: int, end: int, phrase: str, fn: Resolver) -> None:
        if any(s < end and start < e for s, e, _, _ in found):
            return   # overlaps a longer phrase matched earlier ("the day before yesterday" vs "yesterday")
        found.append((start, end, phrase, fn))

    for m in EN_AGO.finditer(text):
        n = EN_NUMBER.get(m.group("n").lower()) or (int(m.group("n")) if m.group("n").isdigit() else None)
        if n is not None and n <= 100:
            take(m.start(), m.end(), m.group(0), _by_unit(EN_UNIT[m.group("u").lower()], -n))
    for m in EN_AHEAD.finditer(text):
        raw, unit = (m.group("n") or m.group("n2")), (m.group("u") or m.group("u2"))
        n = EN_NUMBER.get(raw.lower()) or (int(raw) if raw.isdigit() else None)
        if n is not None and n <= 100:
            take(m.start(), m.end(), m.group(0), _by_unit(EN_UNIT[unit.lower()], n))
    for rx, fn in EN_FIXED_RE:
        for m in rx.finditer(text):
            take(m.start(), m.end(), m.group(0), fn)
    for m in ZH_REL.finditer(text):
        n = cn_number(m.group("n"))
        if n is None or n > 100:
            continue   # "2020年前" means "before 2020", not "2020 years ago"
        sign = -1 if m.group("dir") in ("前", "以前", "之前") else 1
        take(m.start(), m.end(), m.group(0), _by_unit(CN_UNIT[m.group("u")], sign * n))
    for rx, fn in ZH_FIXED_RE:
        for m in rx.finditer(text):
            take(m.start(), m.end(), m.group(0), fn)

    found.sort(key=lambda item: item[0])
    notes: list[str] = []
    seen: set[str] = set()
    for _, _, phrase, fn in found:
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        notes.append(f"{phrase} = {fn(base)}")
        if len(notes) >= MAX_NOTES:
            break
    return notes


def notes_suffix(text: str, event_ms: Optional[int]) -> str:
    notes = relative_time_notes(text, event_ms)
    return " (dates: " + "; ".join(notes) + ")" if notes else ""

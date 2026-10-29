"""Relative-time notes (v0.4): resolved from the turn's own timestamp, English and Chinese, render-time only."""
from datetime import datetime, timezone

from app.timex import cn_number, notes_suffix, relative_time_notes

# Sunday 2025-04-27 07:50 UTC
BASE = int(datetime(2025, 4, 27, 7, 50, tzinfo=timezone.utc).timestamp() * 1000)


def test_english_day_phrases():
    assert relative_time_notes("Did my first 5k yesterday along the river.", BASE) == ["yesterday = 2025-04-26"]
    assert relative_time_notes("We met the day before yesterday, not yesterday.", BASE) == [
        "the day before yesterday = 2025-04-25", "yesterday = 2025-04-26"]
    assert relative_time_notes("See you tomorrow, or the day after tomorrow.", BASE) == [
        "tomorrow = 2025-04-28", "the day after tomorrow = 2025-04-29"]


def test_english_week_month_year_and_ago():
    assert relative_time_notes("last week you suggested a prawn recipe", BASE) == ["last week = week of 2025-04-14"]
    assert relative_time_notes("I paid the deposit three weeks ago and I move next month.", BASE) == [
        "three weeks ago = week of 2025-03-31", "next month = 2025-05"]
    assert relative_time_notes("We adopted him 2 years ago; last year was hard; in 10 days it is over.", BASE) == [
        "2 years ago = 2023", "last year = 2024", "in 10 days = 2025-05-07"]
    assert relative_time_notes("a couple of months ago", BASE) == ["a couple of months ago = 2025-02"]


def test_chinese_phrases():
    assert relative_time_notes("前天刚把样品寄出去，定金是三周前付的。", BASE) == [
        "前天 = 2025-04-25", "三周前 = week of 2025-03-31"]
    assert relative_time_notes("两个月前搬到杭州，明年打算回上海。", BASE) == ["两个月前 = 2025-02", "明年 = 2026"]
    assert relative_time_notes("十五天后发货，上个月的订单已经结清。", BASE) == ["十五天后 = 2025-05-12", "上个月 = 2025-03"]
    assert relative_time_notes("2020年前就认识了", BASE) == []   # "before 2020", not "2020 years ago"


def test_chinese_numerals():
    assert [cn_number(x) for x in ("三", "十", "十五", "二十", "二十三", "两", "7", "百")] == [3, 10, 15, 20, 23, 2, 7, None]


def test_limits_and_absence():
    assert relative_time_notes("yesterday", None) == []
    assert relative_time_notes("", BASE) == []
    many = "yesterday, tomorrow, last week, next week, last year"
    assert len(relative_time_notes(many, BASE)) == 4          # capped
    assert relative_time_notes("today, tonight, this week, this month, this year, 今天, 本周, 今年", BASE) == []   # stamp already says so
    assert relative_time_notes("Yesterday and YESTERDAY again", BASE) == ["Yesterday = 2025-04-26"]   # de-duplicated
    assert notes_suffix("no time words here", BASE) == ""
    assert notes_suffix("昨天到的", BASE) == " (dates: 昨天 = 2025-04-26)"


def test_month_end_clamping():
    jan31 = int(datetime(2025, 1, 31, tzinfo=timezone.utc).timestamp() * 1000)
    assert relative_time_notes("in one month", jan31) == ["in one month = 2025-02"]
    assert relative_time_notes("一个月后", jan31) == ["一个月后 = 2025-02"]

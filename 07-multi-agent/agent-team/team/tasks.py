"""Задачи из бэклога «Пингвин.Хост» с приёмочными тестами (уроки 7.1–7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodingTask:
    name: str
    text: str
    tests: list = field(default_factory=list)
    error_cases: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.tests) + len(self.error_cases)


NORMALIZE_PHONE = CodingTask(
    name="normalize_phone",
    text="""Функция Python normalize_phone(text: str) -> str: приводит российский
номер телефона к каноничному виду - строка из 11 цифр БЕЗ разделителей,
начинается с 7. Вход бывает в разнобой: "+7 (912) 345-67-89", "8-912-345-67-89",
с пробелами. Номер, начинающийся с 8, - это тот же номер с кодом 7.
Если после очистки не получается корректный номер (11 цифр, первая 7 или 8) -
подними ValueError.""",
    tests=[
        ('normalize_phone("+7 (912) 345-67-89")', "79123456789"),
        ('normalize_phone("8-912-345-67-89")', "79123456789"),
        ('normalize_phone("79123456789")', "79123456789"),
        ('normalize_phone("8 (495) 123 45 67")', "74951234567"),
    ],
    error_cases=['normalize_phone("12345")', 'normalize_phone("нет телефона")'],
)

PARSE_SIZE = CodingTask(
    name="parse_size",
    text="""Функция Python parse_size(text: str) -> int: переводит человекочитаемый
размер диска в число байтов (множитель 1024). Единицы: КБ, МБ, ГБ, ТБ (русскими
буквами). Пробел между числом и единицей необязателен. Число может быть дробным.
Голое число без единицы - это уже байты.""",
    tests=[
        ('parse_size("512 МБ")', 536870912),
        ('parse_size("1.5 ГБ")', 1610612736),
        ('parse_size("2 ТБ")', 2199023255552),
        ('parse_size("100КБ")', 102400),
        ('parse_size("0.5 КБ")', 512),
        ('parse_size("777")', 777),
    ],
)

TASKS = {task.name: task for task in (NORMALIZE_PHONE, PARSE_SIZE)}

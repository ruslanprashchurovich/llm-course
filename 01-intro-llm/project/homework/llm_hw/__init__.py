"""llm_hw — эталонное решение «Заданий для прокачки» из урока 1.8.

Расширение финального проекта модуля 1: те же пять команд плюс пять доработок
из домашнего задания. CLI переписан на typer, вывод — на rich (бонус задания 5).

Карта заданий → где живёт решение:

    1. --stats везде + облачная цена   ->  pricing.py, stats.py, llmio.py
    2. команда commit                  ->  cli.py: cmd `commit`
    3. команда translate с глоссарием  ->  cli.py: cmd `translate`
    4. персистентный чат               ->  history.py, cli.py: cmd `chat`
    5. упаковка (pyproject + entry)    ->  pyproject.toml, cli.py: main()
"""

__version__ = "0.1.0"

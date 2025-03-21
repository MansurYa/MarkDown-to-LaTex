**Автоматическое преобразование Markdown в LaTeX с использованием GPT-4o-mini**

[defd]

---

## 📌 Основные возможности

- Полная поддержка русского языка через XeLaTeX
- Автоматическая генерация титульной страницы
- Преобразование сложных элементов:
  - Таблицы с адаптивным форматированием
  - Блоки кода с подсветкой синтаксиса
  - Математические формулы и изображения
  - Иерархические списки любой вложенности
- Параллельная обработка больших текстов

## 🚀 Быстрый старт

### Требования
- Python 3.7+
- OpenAI API ключ
- Установленные пакеты: `requests`, `tqdm`

```bash
# Установка зависимостей
pip install -r requirements.txt
```

### Конфигурация
Заполните `config.json`:
```json
{
    "api_key": "ваш_openai_api_ключ", 
    "organization": "ваш_openai_organization_ключ",
    "model_name_for_conversion": "gpt-4o-mini",
    "max_symbols_block_size_for_conversation_MarkDown_to_LaTex": 20000
}
```

### Запуск
```bash
python3 MarkDown_to_LaTex.py --input report.md --output thesis.tex --title --org "МГУ" --author "Иванов И.И."
```

### Использование как модуля
```python
from MarkDown_to_LaTex import MarkDown_to_LaTex

latex_output = MarkDown_to_LaTex(
    markdown_text="# Ваше исследование\n\nОсновной текст...",
    organization_name="Санкт-Петербургский государственный университет.",
    organization_section_name="Направление 'Большие данные и распределенная цифровая платформа'",
    type_of_work_name="Отчёт по проекту",
    work_name="Разработка асинхронного чат-сервера",
    persons_who_completed_the_work_full_name=["Иванов Иван Иванович", "Скороходов Андрей Маратович"],
    team_name="Группа: 22.Б15-пу",
    bosses_full_name=["Серая Ольга Петровна"],
    city_name="Москва",
    year_str="2025"
)
```

### Компиляция в pdf

Для преобразования сгенерированного LaTeX-кода в PDF рекомендуем использовать онлайн-редактор Overleaf:

1. Перейдите на [Overleaf](https://www.overleaf.com) и авторизуйтесь
2. Создайте новый проект → **Blank Project**
3. В левом меню проекта:
   - Нажмите ⚙️ **Settings**
   - В разделе *Compiler* выберите **XeLaTeX**
4. Замените содержимое `main.tex` вашим кодом
5. Добавьте необходимые ресурсы (изображения, библиографию)
6. Нажмите **Recompile** для сборки документа
7. Скачайте готовый PDF через меню ⋯ → **Download PDF**

## 🎯 Пример преобразования

**Входной Markdown**:
```markdown
## Результаты эксперимента

| Параметр    | Значение |
|-------------|----------|
| Точность    | 92.3%    |
| Время обработки | 2.4 сек |
```

**Выходной LaTeX**:
```latex
\newpage
\newsection{Результаты эксперимента}

\begin{table}[h!]
    \centering
    \begin{tabular}{|l|l|}
        \hline
        \textbf{Параметр} & \textbf{Значение} \\ 
        \hline
        Точность & 92.3\% \\
        \hline
        Время обработки & 2.4 сек \\
        \hline
    \end{tabular}
    \caption*{Результаты эксперимента}
\end{table}
```

Для демонстрации работы конвертера ознакомьтесь с файлом [example_of_conversion_to_LaTeX.pdf](example_of_conversion_to_LaTeX.pdf) - это скомпилированный PDF-документ, полученный из LaTeX-кода, который был сгенерирован скриптом `example_MarkDown_to_LaTex.py`.

🔗 Основано на проекте [Chat GPT with chain of reasoning](https://github.com/MansurYa/chat_GPT_with_chain_of_reasoning.git)

import concurrent.futures
import json
from typing import List
from chat_GPT_manager import ChatGPTAgent

try:
    with open("../config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Файл config.json не найден.")
    exit(1)
except json.JSONDecodeError:
    print("Ошибка при чтении config.json.")
    exit(1)

MAX_SYMBOLS_BLOCK_SIZE = config.get("max_symbols_block_size_for_conversation_MarkDown_to_LaTex", 20000)
MODEL_NAME_FOR_CONVERSION = config.get("model_name_for_conversion", "gpt-4o-mini")
API_KEY = config["api_key"]
ORGANIZATION = config["organization"]
TEMPERATURE = 0.0
MAX_RESPONSE_TOKENS = 10000
MAX_TOTAL_TOKENS = 50000

try:
    with open("../prompts/prompt_MarkDown_to_LaTex.txt", "r", encoding="utf-8") as f:
        PROMPT = f.read()
except FileNotFoundError:
    print("Файл prompt_MarkDown_to_LaTex.txt не найден.")
    exit(1)

try:
    with open("../prompts/LaTex_title_page.txt", "r", encoding="utf-8") as f:
        TITLE_PAGE = f.read()
except FileNotFoundError:
    print("Файл LaTex_title_page.txt не найден.")
    exit(1)


def __split_text(text: str) -> List[str]:
    """
    Разделяет текст на блоки, каждый из которых не превышает MAX_SYMBOLS_BLOCK_SIZE символов.

    :param text: Входной текст в формате Markdown
    :return: Список строк, каждая из которых является блоком текста
    """
    HEADINGS = ["###### ", "##### ", "#### ", "### ", "## ", "# "]

    def split_by_heading(block: str, heading_token: str) -> List[str]:
        """
        Делит блок по строкам, начинающимся с heading_token, включая заголовок в новый блок.

        :param block: Текст блока
        :param heading_token: Токен заголовка
        :return: Список подблоков
        """
        lines = block.splitlines(keepends=True)
        result_blocks = []
        current_block = []

        for line in lines:
            if line.lstrip().startswith(heading_token):
                if current_block:
                    result_blocks.append("".join(current_block))
                    current_block = []
                current_block.append(line)
            else:
                current_block.append(line)

        if current_block:
            result_blocks.append("".join(current_block))

        return result_blocks

    def split_by_half_approx(block: str) -> List[str]:
        """
        Разделяет блок примерно пополам, стараясь разрезать по пробелу.

        :param block: Текст блока
        :return: Список разделенных частей
        """
        if len(block) <= MAX_SYMBOLS_BLOCK_SIZE:
            return [block]
        mid = len(block) // 2
        cut_point = None
        left, right = mid, mid
        while left >= 0 or right < len(block):
            if left >= 0 and block[left].isspace():
                cut_point = left
                break
            if right < len(block) and block[right].isspace():
                cut_point = right
                break
            left -= 1
            right += 1
        cut_point = cut_point if cut_point is not None else mid
        part1, part2 = block[:cut_point], block[cut_point:]
        return split_by_half_approx(part1) + split_by_half_approx(part2)

    def split_recursively(block: str, heading_index: int = 0) -> List[str]:
        """
        Рекурсивно разделяет блок по заголовкам или пополам.

        :param block: Текст блока
        :param heading_index: Индекс текущего уровня заголовка
        :return: Список блоков
        """
        if len(block) <= MAX_SYMBOLS_BLOCK_SIZE:
            return [block]
        if heading_index < len(HEADINGS):
            token = HEADINGS[heading_index]
            splitted_temp = split_by_heading(block, token)
            if len(splitted_temp) > 1:
                result_blocks = []
                for sub_block in splitted_temp:
                    if len(sub_block) > MAX_SYMBOLS_BLOCK_SIZE:
                        result_blocks.extend(split_recursively(sub_block, heading_index + 1))
                    else:
                        result_blocks.append(sub_block)
                return result_blocks
            return split_recursively(block, heading_index + 1)
        return split_by_half_approx(block)

    def merge_blocks(blocks: List[str]) -> List[str]:
        """
        Объединяет блоки, если их можно объединить без превышения MAX_SYMBOLS_BLOCK_SIZE.

        :param blocks: Список блоков
        :return: Список объединенных блоков
        """
        merged = []
        buffer_block = ""
        for b in blocks:
            if not buffer_block:
                buffer_block = b
            elif len(buffer_block) + len(b) <= MAX_SYMBOLS_BLOCK_SIZE:
                buffer_block += b
            else:
                merged.append(buffer_block)
                buffer_block = b
        if buffer_block:
            merged.append(buffer_block)
        return merged

    raw_blocks = split_recursively(text, 0)
    return merge_blocks(raw_blocks)


def __convert_MarkDown_to_LaTex_for_block(markdown_text_block: str) -> str:
    """
    Создает экземпляр ChatGPTAgent и преобразует блок Markdown в LaTeX.

    :param markdown_text_block: Блок текста в формате Markdown
    :return: Строка в формате LaTeX для этого блока
    """
    try:
        agent = ChatGPTAgent(
            api_key=API_KEY,
            organization=ORGANIZATION,
            model_name=MODEL_NAME_FOR_CONVERSION,
            mode=1,
            task_prompt=PROMPT,
            max_total_tokens=MAX_TOTAL_TOKENS,
            max_response_tokens=MAX_RESPONSE_TOKENS,
            temperature=TEMPERATURE
        )
        return agent.response_from_chat_GPT(markdown_text_block)
    except Exception as e:
        print(f"Ошибка в __convert_MarkDown_to_LaTex_for_block: {e}")
        return ""


def MarkDown_to_LaTex(
    MarkDown_text: str,
    organization_name: str = "",
    organization_section_name: str = "",
    type_of_work_name: str = "",
    work_name: str = "",
    persons_who_completed_the_work_full_name: List[str] = None,
    team_name: str = "",
    bosses_full_name: List[str] = None,
    city_name: str = "",
    year_str: str = ""
) -> str:
    """
    Преобразует весь текст Markdown в LaTeX, добавляя титульную страницу, если указаны параметры.

    :param MarkDown_text: Текст в формате Markdown
    :param organization_name: Название организации
    :param organization_section_name: Название отдела организации
    :param type_of_work_name: Тип работы
    :param work_name: Название работы
    :param persons_who_completed_the_work_full_name: Список полных имен исполнителей
    :param team_name: Название команды
    :param bosses_full_name: Список полных имен руководителей
    :param city_name: Название города
    :param year_str: Год в виде строки
    :return: Строка в формате LaTeX
    """
    MarkDown_text_blocks = __split_text(MarkDown_text)

    def persons_name_to_needed_format(names: List[str]) -> str:
        """
        Преобразует список имен в формат, подходящий для LaTeX.

        :param names: Список имен
        :return: Строка с именами, разделенными запятыми и переносами строк
        """
        output = ""
        for name in names:
            output += f"{name},\\\\\n                "
        last_comma_index = output.rfind(',')
        if last_comma_index != -1:
            return output[:last_comma_index] + output[last_comma_index + 1:]
        return output

    title = ""
    if any([organization_name, organization_section_name, type_of_work_name, work_name,
            persons_who_completed_the_work_full_name, team_name, bosses_full_name,
            city_name, year_str]):
        persons_who_completed = persons_who_completed_the_work_full_name or []
        bosses_names = bosses_full_name or []
        title = (
            TITLE_PAGE
            .replace("organization_name", organization_name)
            .replace("organization_section_name", organization_section_name)
            .replace("type_of_work_name", type_of_work_name)
            .replace("work_name", work_name)
            .replace("persons_who_completed_the_work_full_name",
                     persons_name_to_needed_format(persons_who_completed))
            .replace("team_name", team_name)
            .replace("bosses_full_name", persons_name_to_needed_format(bosses_names))
            .replace("city_name", city_name)
            .replace("year_str", year_str)
        )

    converted_blocks = ["" for _ in range(len(MarkDown_text_blocks))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(__convert_MarkDown_to_LaTex_for_block, block): i
            for i, block in enumerate(MarkDown_text_blocks)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            block_index = future_to_index[future]
            try:
                converted_blocks[block_index] = future.result()
            except Exception as exc:
                print(f"Блок {block_index} не обработан: {exc}")
                converted_blocks[block_index] = ""

    return title + "\n" + "".join(converted_blocks) + "\n\\end{document}"


def main():
    input_path = input("Введите путь к файлу Markdown: ")
    output_path = input("Введите путь для сохранения файла LaTeX: ")

    include_title = input("Вы хотите включить титульный лист? (да/нет): ").lower()

    if include_title == "yes":
        organization_name = input("Enter organization name: ").strip()
        organization_section_name = input("Введите название раздела организации: ").strip()
        type_of_work_name = input("Введите тип работы: ").strip()
        work_name = input("Введите название работы: ").strip()

        print("Введите имена людей, выполнивших работу, по одному в каждой строке. По завершении введите 'done'.")
        persons = []
        while True:
            name = input().strip()
            if name.lower() == "done":
                break
            if name:
                persons.append(name)

        team_name = input("Введите название группы/команды: ").strip()

        print("Введите полные имена боссов, по одному на строку. Когда закончите, введите 'done'.")
        bosses = []
        while True:
            name = input().strip()
            if name.lower() == "done":
                break
            if name:
                bosses.append(name)

        city_name = input("Введите название города: ").strip()
        year_str = input("Введите год: ").strip()
    else:
        organization_name = ""
        organization_section_name = ""
        type_of_work_name = ""
        work_name = ""
        persons = []
        team_name = ""
        bosses = []
        city_name = ""
        year_str = ""

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        print("Обработка... Это может занять некоторое время.")

        latex_text = MarkDown_to_LaTex(
            markdown_text,
            organization_name,
            organization_section_name,
            type_of_work_name,
            work_name,
            persons,
            team_name,
            bosses,
            city_name,
            year_str
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_text)

        # Success message
        print("Конвертация прошла успешно! Файл LaTeX сохранен в", output_path)

    except FileNotFoundError:
        print("Error: Входной файл не найден.")
    except PermissionError:
        print("Error: Отказано в разрешении на запись в выходной файл.")
    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    main()

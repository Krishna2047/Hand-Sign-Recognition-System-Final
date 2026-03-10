import os
import json

BASE_DIR = os.path.dirname(__file__)
CONV_DIR = os.path.join(BASE_DIR, "conversations")

os.makedirs(CONV_DIR, exist_ok=True)


def save_conversation(user, gesture, text, language):

    file_path = os.path.join(CONV_DIR, f"{user}.json")

    data = []

    # safely load existing data
    if os.path.exists(file_path):

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = []

    data.append({
        "gesture": gesture,
        "text": text,
        "language": language
    })

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_history(user):

    file_path = os.path.join(CONV_DIR, f"{user}.json")

    if not os.path.exists(file_path):
        return []

    try:

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except:

        return []
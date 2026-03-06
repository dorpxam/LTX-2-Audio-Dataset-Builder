import json
from pathlib import Path

class Translator:
    def __init__(self):
        self.language: str = "en"
        self.strings: dict[str, dict[str, str]] = {}
        self._load_strings()

    def _load_strings(self):
        json_path = Path(__file__).parent.parent / "config" / "translation.json"
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.strings = json.load(f)
        except Exception as e:
            print(f"Critical: Could not load translation.json: {e}")
            self.strings = {"en": {}, "fr": {}}

    def set_language(self, lang: str) -> None:
        if lang in self.strings:
            self.language = lang

    def translate(self, key: str, **kwargs: str) -> str:
        text = self.strings.get(self.language, {}).get(key, key)
        try:
            return text.format(**kwargs)
        except KeyError as e:
            return f"{text} (Missing arg: {e})"

T = Translator()
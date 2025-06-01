# ./app/utils.py

import logging
import nltk
from rapidfuzz import process, fuzz
from data.config import CONFIG
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)


# Загрузка тонального словаря
def load_tonal_dict():
    tonal_dict = {}
    try:
        with open('data/tonal_dict.txt', encoding='utf-8') as f:
            for line in f:
                word, score = line.strip().split('\t')
                tonal_dict[word] = float(score)
    except FileNotFoundError:
        logger.error("Файл tonal_dict.txt не найден")
    return tonal_dict


TONAL_DICT = load_tonal_dict()


# Очистка фразы
def clear_phrase(phrase):
    if not phrase:
        return ""
    phrase = phrase.lower()
    alphabet = '1234567890qwertyuiopasdfghjklzxcvbnmабвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    return ''.join(symbol for symbol in phrase if symbol in alphabet).strip()


# Лемматизация и морфологический анализ
def lemmatize_phrase(phrase):
    if not phrase:
        return ""
    cleaned_phrase = clear_phrase(phrase)
    if not cleaned_phrase:
        return ""
    doc = Doc(cleaned_phrase)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    lemmatized_words = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = token.lemma if token.lemma else token.text
        lemmatized_words.append(lemma)
    return ' '.join(lemmatized_words)


# Анализ тональности
def analyze_sentiment(phrase):
    if not phrase:
        return 'neutral'
    lemmatized = lemmatize_phrase(phrase)
    words = lemmatized.split()
    sentiment_score = 0
    count = 0
    for word in words:
        if word in TONAL_DICT:
            sentiment_score += TONAL_DICT[word]
            count += 1
    if count == 0:
        return 'neutral'
    avg_score = sentiment_score / count
    if avg_score > 0.3:
        return 'positive'
    elif avg_score < -0.3:
        return 'negative'
    return 'neutral'


# Проверка на осмысленность текста
def is_meaningful_text(text):
    text = clear_phrase(text)
    words = text.split()
    return any(len(word) > 2 and all(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in word) for word in words)


# Извлечение возраста
def extract_age(replica):
    replica = lemmatize_phrase(replica)
    logger.info(f"Extracting age from: '{replica}'")
    words = replica.split()
    for i, word in enumerate(words):
        if word.isdigit() and (i + 1 < len(words) and words[i + 1] in ['год', 'года', 'лет'] or 'для' in words[:i]):
            logger.info(f"Found age: {word}")
            return word
    logger.info("Age not found")
    return None


# Извлечение цены
def extract_price(replica):
    replica = clear_phrase(replica)
    logger.info(f"Extracting price from: '{replica}'")
    if not replica:
        return None
    words = replica.split()
    for i, word in enumerate(words):
        if word.isdigit() and (
                i + 1 < len(words) and words[i + 1] in ['рублей', 'руб'] or 'до' in words[:i] or 'дешевле' in words[
                                                                                                              :i]):
            logger.info(f"Found price: {word}")
            return int(word)
    logger.info("Price not found")
    return None


# Извлечение игрушки
def extract_toy_name(replica):
    replica = lemmatize_phrase(replica)
    if not replica:
        return None
    # Проверяем точное совпадение с названиями игрушек
    for toy in CONFIG['toys'].keys():
        toy_lemmatized = lemmatize_phrase(toy)
        if toy_lemmatized in replica:
            return toy
    # Проверяем синонимы и нечёткое соответствие
    for toy, data in CONFIG['toys'].items():
        synonyms_lemmatized = [lemmatize_phrase(syn) for syn in data.get('synonyms', [])]
        if any(syn in replica for syn in synonyms_lemmatized):
            return toy
        candidates = [toy] + data.get('synonyms', [])
        best_match = process.extractOne(replica, candidates, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] > CONFIG['thresholds']['fuzzy_match_toy']:
            return toy
    # Специальная обработка для пазлов с числом элементов
    words = replica.split()
    for i, word in enumerate(words):
        if word.isdigit() and (i + 1 < len(words) and words[i + 1] in ['элемент', 'элементов']) and 'пазл' in words:
            puzzle_name = f"Пазл {word} элементов"
            if puzzle_name in CONFIG['toys']:
                return puzzle_name
    return None


# Извлечение категории
def extract_toy_category(replica):
    replica = lemmatize_phrase(replica)
    if not replica:
        return None
    for toy, data in CONFIG['toys'].items():
        for category in data.get('categories', []):
            category_lemmatized = lemmatize_phrase(category)
            category_synonyms = data.get('category_synonyms', {}).get(category, [])
            synonyms_lemmatized = [lemmatize_phrase(syn) for syn in category_synonyms]
            if category_lemmatized in replica or any(syn in replica for syn in synonyms_lemmatized):
                return category
    return None


# Проверка возраста в диапазоне
def is_age_in_range(age, age_range):
    try:
        age = int(age)
        min_age = age_range['min_age']
        max_age = age_range['max_age']
        return min_age <= age <= (max_age if max_age is not None else float('inf'))
    except (ValueError, TypeError):
        return False


# Класс для управления статистикой
class Stats:
    def __init__(self, context):
        self.context = context
        if 'stats' not in context.user_data:
            context.user_data['stats'] = {'intent': 0, 'generate': 0, 'failure': 0}
        self.stats = context.user_data['stats']

    def add(self, type, replica, answer, context):
        """Обновляет статистику, сохраняет её в context и логирует."""
        if type in self.stats:
            self.stats[type] += 1
        else:
            self.stats[type] = 1
        self.context.user_data['stats'] = self.stats
        logger.info(f"Stats: {self.stats} | Вопрос: {replica} | Ответ: {answer}")

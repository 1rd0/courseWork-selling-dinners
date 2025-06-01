# ./app/train_dialogues_model.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clear_phrase, lemmatize_phrase, logger

logger.info("Начинается обучение модели для dialogues.txt")

# Загрузка dialogues.txt
dialogues = []
try:
    with open('data/dialogues.txt', encoding='utf-8') as f:
        content = f.read()
    dialogues = [d.split('\n')[:2] for d in content.split('\n\n') if len(d.split('\n')) >= 2]
    dialogues = [(q[1:].strip() if q.startswith('-') else q, a[1:].strip() if a.startswith('-') else a) for q, a in
                 dialogues]
except Exception as e:
    logger.error(f"Ошибка чтения dialogues.txt: {e}")
    exit(1)

# Подготовка данных
questions = [lemmatize_phrase(clear_phrase(q)) for q, _ in dialogues]
answers = [a for _, a in dialogues]

# Обучение TF-IDF модели
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(questions)

# Сохранение модели
with open('models/dialogues_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('models/dialogues_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
with open('models/dialogues_answers.pkl', 'wb') as f:
    pickle.dump(answers, f)

logger.info("Модель для dialogues.txt обучена и сохранена в ./models/")

# ./app/bot.py

import random
import pickle
import os
import logging
import traceback
from enum import Enum
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from data.config import CONFIG
from sklearn.metrics.pairwise import cosine_similarity
from utils import clear_phrase, is_meaningful_text, extract_dish_name, extract_dish_category, extract_price, \
    Stats, logger, lemmatize_phrase, analyze_sentiment
from rapidfuzz import process, fuzz

# Загрузка токена
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Состояния бота
class BotState(Enum):
    NONE = "NONE"
    WAITING_FOR_DISH = "WAITING_FOR_DISH"
    WAITING_FOR_INTENT = "WAITING_FOR_INTENT"

# Намерения
class Intent(Enum):
    HELLO = "hello"
    BYE = "bye"
    YES = "yes"
    NO = "no"
    DISH_TYPES = "dish_types"
    DISH_PRICE = "dish_price"
    DISH_AVAILABILITY = "dish_availability"
    DISH_RECOMMENDATION = "dish_recommendation"
    FILTER_DISHES = "filter_dishes"
    DISH_INFO = "dish_info"
    ORDER_DISH = "order_dish"
    COMPARE_DISHES = "compare_dishes"

# Типы ответов
class ResponseType(Enum):
    INTENT = "intent"
    GENERATE = "generate"
    FAILURE = "failure"

# Класс бота
class Bot:
    def __init__(self):
        """Инициализация моделей."""
        try:
            with open('models/intent_model.pkl', 'rb') as f:
                self.clf = pickle.load(f)
            with open('models/intent_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('models/dialogues_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            with open('models/dialogues_matrix.pkl', 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open('models/dialogues_answers.pkl', 'rb') as f:
                self.answers = pickle.load(f)
        except FileNotFoundError as e:
            logger.error(f"Не найдены файлы модели: {e}\n{traceback.format_exc()}")
            raise

    def _update_context(self, context, replica, answer, intent=None):
        """Обновляет контекст пользователя."""
        context.user_data.setdefault('state', BotState.NONE.value)
        context.user_data.setdefault('current_dish', None)
        context.user_data.setdefault('last_bot_response', None)
        context.user_data.setdefault('last_intent', None)
        context.user_data.setdefault('history', [])

        context.user_data['history'].append(replica)
        context.user_data['history'] = context.user_data['history'][-CONFIG['history_limit']:]
        context.user_data['last_bot_response'] = answer
        if intent:
            context.user_data['last_intent'] = intent

    def classify_intent(self, replica):
        """Классифицирует намерение пользователя."""
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized:
            return None
        vectorized = self.vectorizer.transform([replica_lemmatized])
        intent = self.clf.predict(vectorized)[0]
        best_score = 0
        best_intent = None
        for intent_key, data in CONFIG['intents'].items():
            examples = [lemmatize_phrase(ex) for ex in data.get('examples', []) if lemmatize_phrase(ex)]
            if not examples:
                continue
            match = process.extractOne(replica_lemmatized, examples, scorer=fuzz.ratio)
            if match and match[1] / 100 > best_score and match[1] / 100 >= CONFIG['thresholds']['intent_score']:
                best_score = match[1] / 100
                best_intent = intent_key
        logger.info(
            f"Classify intent: replica='{replica_lemmatized}', predicted='{intent}', best_intent='{best_intent}', score={best_score}")
        return best_intent or intent if best_score >= CONFIG['thresholds']['intent_score'] else None

    def _get_dish_response(self, intent, dish_name, replica, context):
        """Обрабатывает запросы, связанные с конкретным блюдом."""
        if dish_name not in CONFIG['dishes']:
            return "Извините, такого блюда нет в меню."
        responses = CONFIG['intents'][intent]['responses']
        answer = random.choice(responses)
        dish_data = CONFIG['dishes'][dish_name]
        answer = answer.replace('[dish_name]', dish_name)
        answer = answer.replace('[price]', str(dish_data['price']))
        answer = answer.replace('[description]', dish_data.get('description', 'вкусное блюдо'))

        # Добавляем реакцию на тональность
        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Рад, что вы в хорошем настроении! 😊"
        elif sentiment == 'negative':
            answer += " Кажется, вы не в духе. Может, вкусное блюдо поднимет настроение? 😊"

        return f"{answer} Что ещё интересует?"

    def _find_dish_by_context(self, replica, context):
        """Ищет блюдо на основе контекста или категории."""
        last_response = context.user_data.get('last_bot_response', '')
        last_intent = context.user_data.get('last_intent', '')
        dish_category = extract_dish_category(replica)

        if last_response and 'Кстати, у нас есть' in last_response:
            return extract_dish_name(last_response)
        elif dish_category:
            suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if dish_category in data.get('categories', [])]
            return random.choice(suitable_dishes) if suitable_dishes else None
        elif last_intent == Intent.DISH_TYPES.value:
            for hist in context.user_data.get('history', [])[::-1]:
                hist_dish = extract_dish_name(hist)
                if hist_dish:
                    return hist_dish
                hist_category = extract_dish_category(hist)
                if hist_category:
                    suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if
                                       hist_category in data.get('categories', [])]
                    if suitable_dishes:
                        return random.choice(suitable_dishes)
        return None

    def _handle_filter_dishes(self, price, dish_category, context):
        """Обрабатывает фильтрацию блюд по цене и категории."""
        suitable_dishes = [
            dish for dish, data in CONFIG['dishes'].items()
            if (not price or data['price'] <= price)
               and (not dish_category or dish_category in data.get('categories', []))
        ]
        recent_dishes = [extract_dish_name(h) for h in context.user_data.get('history', [])]
        suitable_dishes = [d for d in suitable_dishes if d not in recent_dishes]

        if not suitable_dishes:
            conditions = []
            if price:
                conditions.append(f"до {price} рублей")
            if dish_category:
                conditions.append(f"в категории {dish_category}")
            return f"Извините, нет блюд для {', '.join(conditions)}."

        dishes_list = ', '.join(suitable_dishes)
        if not price and not dish_category:
            dish_name = random.choice(suitable_dishes)
            context.user_data['current_dish'] = dish_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            return f"Советую {dish_name}! Хотите узнать цену или состав?"
        return f"Вот что нашлось: {dishes_list}."

    def get_answer_by_intent(self, intent, replica, context):
        """Генерирует ответ на основе намерения."""
        dish_name = context.user_data.get('current_dish')
        last_intent = context.user_data.get('last_intent', '')
        dish_category = extract_dish_category(replica)
        price = extract_price(replica)

        if intent not in CONFIG['intents']:
            return None
        responses = CONFIG['intents'][intent]['responses']
        if not responses:
            return None
        answer = random.choice(responses)

        # Добавляем реакцию на тональность
        sentiment = analyze_sentiment(replica)
        sentiment_suffix = ""
        if sentiment == 'positive':
            sentiment_suffix = " Рад, что вы в хорошем настроении! 😊"
        elif sentiment == 'negative':
            sentiment_suffix = " Кажется, вы не в духе. Давайте подберём что-то вкусное! 😊"

        if intent in [Intent.DISH_PRICE.value, Intent.DISH_AVAILABILITY.value, Intent.DISH_INFO.value,
                      Intent.ORDER_DISH.value]:
            if not dish_name:
                dish_name = self._find_dish_by_context(replica, context)
                if dish_name:
                    context.user_data['current_dish'] = dish_name
                    context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                    return f"Из {dish_category or 'блюд'} есть {dish_name}. Хотите узнать цену, состав или наличие?{sentiment_suffix}"
                context.user_data['state'] = BotState.WAITING_FOR_DISH.value
                return f"Какое блюдо или категорию вы имеете в виду?{sentiment_suffix}"
            return self._get_dish_response(intent, dish_name, replica, context)

        elif intent == Intent.DISH_RECOMMENDATION.value:
            answer = self._handle_filter_dishes(None, dish_category, context)

        elif intent == Intent.FILTER_DISHES.value:
            if price or dish_category:
                answer = self._handle_filter_dishes(price, dish_category, context)
            else:
                return f"Укажите цену или категорию для фильтрации.{sentiment_suffix}"

        elif intent == Intent.DISH_TYPES.value:
            categories = random.sample([cat for dish in CONFIG['dishes'].values() for cat in dish.get('categories', [])],
                                       min(3, len(CONFIG['dishes'])))
            dishes = random.sample(list(CONFIG['dishes'].keys()), min(2, len(CONFIG['dishes'])))
            answer = f"У нас есть {', '.join(set(categories))} и блюда вроде {', '.join(dishes)}. Что интересно?{sentiment_suffix}"
            context.user_data['current_dish'] = None

        elif intent == Intent.COMPARE_DISHES.value:
            dish1 = random.choice(list(CONFIG['dishes'].keys()))
            dish2 = random.choice([d for d in CONFIG['dishes'].keys() if d != dish1])
            answer = answer.replace('[dish1]', dish1).replace('[dish2]', dish2)
            context.user_data['current_dish'] = dish1
            answer += f" Что интересует: {dish1} или {dish2}?{sentiment_suffix}"

        elif intent == Intent.YES.value:
            if last_intent == Intent.HELLO.value:
                categories = random.sample(
                    [cat for dish in CONFIG['dishes'].values() for cat in dish.get('categories', [])],
                    min(3, len(CONFIG['dishes'])))
                answer = f"Отлично! У нас есть {', '.join(set(categories))}. Что хотите узнать?{sentiment_suffix}"
            elif last_intent in [Intent.DISH_PRICE.value, Intent.DISH_INFO.value, Intent.DISH_AVAILABILITY.value,
                                 Intent.ORDER_DISH.value]:
                if dish_name:
                    answer = f"Цена на {dish_name} — {CONFIG['dishes'][dish_name]['price']} рублей. Что ещё интересует?{sentiment_suffix}"
                else:
                    answer = f"Назови блюдо, чтобы я рассказал подробнее!{sentiment_suffix}"
            elif last_intent == Intent.DISH_TYPES.value:
                dishes = random.sample(list(CONFIG['dishes'].keys()), min(2, len(CONFIG['dishes'])))
                answer = f"У нас есть {', '.join(dishes)}. Назови одно, чтобы узнать больше!{sentiment_suffix}"
            elif last_intent == 'offtopic':
                answer = f"Хорошо, давай продолжим! Хочешь узнать про блюда?{sentiment_suffix}"
            else:
                answer = f"Хорошо, что интересует? Блюда, цены или что-то ещё?{sentiment_suffix}"

        elif intent == Intent.NO.value:
            context.user_data['current_dish'] = None
            context.user_data['state'] = BotState.NONE.value
            answer = f"Хорошо, какое блюдо обсудим теперь?{sentiment_suffix}"

        if intent in [Intent.HELLO.value, Intent.DISH_TYPES.value] and random.random() < 0.2:
            ad_dish = random.choice([d for d in CONFIG['dishes'].keys() if d != dish_name])
            answer += f" Кстати, у нас есть {ad_dish} — отличный выбор для обеда!{sentiment_suffix}"

        context.user_data['last_intent'] = intent
        return answer

    def generate_answer(self, replica, context):
        """Генерирует ответ на основе диалогов."""
        replica_lemmatized = lemmatize_phrase(replica)
        if not replica_lemmatized or not self.answers:
            return None
        if not is_meaningful_text(replica):
            return None
        replica_vector = self.tfidf_vectorizer.transform([replica_lemmatized])
        similarities = cosine_similarity(replica_vector, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        if similarities[best_idx] > CONFIG['thresholds']['dialogues_similarity']:
            answer = self.answers[best_idx]
            logger.info(
                f"Found in dialogues.txt: replica='{replica_lemmatized}', answer='{answer}', similarity={similarities[best_idx]}")
            # Добавляем реакцию на тональность
            sentiment = analyze_sentiment(replica)
            if sentiment == 'positive':
                answer += " Рад, что ты в хорошем настроении! 😊"
            elif sentiment == 'negative':
                answer += " Кажется, ты не в духе. Может, вкусное блюдо поднимет настроение? 😊"
            if random.random() < 0.3:
                ad_dish = random.choice(list(CONFIG['dishes'].keys()))
                answer += f" Кстати, у нас есть {ad_dish} — отличный выбор для обеда!"
            context.user_data['last_intent'] = 'offtopic'
            return answer
        logger.info(f"No match in dialogues.txt for replica='{replica_lemmatized}'")
        return None

    def get_failure_phrase(self, replica):
        """Возвращает фразу при неудачном запросе с учетом тональности."""
        dish_name = random.choice(list(CONFIG['dishes'].keys()))
        answer = random.choice(CONFIG['failure_phrases']).replace('[dish_name]', dish_name)
        sentiment = analyze_sentiment(replica)
        if sentiment == 'positive':
            answer += " Ты в отличном настроении, давай найдем вкусное блюдо! 😊"
        elif sentiment == 'negative':
            answer += " Не переживай, давай подберем что-то вкусное! 😊"
        return answer

    def _process_none_state(self, replica, context):
        """Обрабатывает состояние NONE."""
        dish_name = extract_dish_name(replica)
        if dish_name:
            context.user_data['current_dish'] = dish_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            sentiment = analyze_sentiment(replica)
            suffix = " Рад, что ты в хорошем настроении! 😊" if sentiment == 'positive' else " Кажется, ты не в духе. Давай найдем что-то вкусное? 😊" if sentiment == 'negative' else ""
            return f"Вы имеете в виду {dish_name}? Хотите узнать цену, состав или наличие?{suffix}"

        dish_category = extract_dish_category(replica)
        if dish_category:
            suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if dish_category in data.get('categories', [])]
            if suitable_dishes:
                dish_name = random.choice(suitable_dishes)
                context.user_data['current_dish'] = dish_name
                context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                sentiment = analyze_sentiment(replica)
                suffix = " Ты в отличном настроении, давай продолжим! 😊" if sentiment == 'positive' else " Не грусти, найдем что-то вкусное! 😊" if sentiment == 'negative' else ""
                return f"Из {dish_category} есть {dish_name}. Хотите узнать цену, состав или наличие?{suffix}"
            sentiment = analyze_sentiment(replica)
            suffix = " В хорошем настроении? Давай попробуем другую категорию! 😊" if sentiment == 'positive' else " Не переживай, попробуем другую категорию! 😊" if sentiment == 'negative' else ""
            return f"У нас нет блюд в категории {dish_category}. Попробуйте другую категорию!{suffix}"

        intent = self.classify_intent(replica)
        if intent:
            return self.get_answer_by_intent(intent, replica, context)

        return self.generate_answer(replica, context) or self.get_failure_phrase(replica)

    def _process_waiting_for_dish(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_DISH."""
        dish_name = extract_dish_name(replica)
        if dish_name:
            context.user_data['current_dish'] = dish_name
            context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
            sentiment = analyze_sentiment(replica)
            suffix = " Отличное настроение, да? 😊" if sentiment == 'positive' else " Давай найдем что-то вкусное! 😊" if sentiment == 'negative' else ""
            return f"Вы имеете в виду {dish_name}? Хотите узнать цену, состав или наличие?{suffix}"
        dish_category = extract_dish_category(replica)
        if dish_category:
            suitable_dishes = [dish for dish, data in CONFIG['dishes'].items() if dish_category in data.get('categories', [])]
            if suitable_dishes:
                dish_name = random.choice(suitable_dishes)
                context.user_data['current_dish'] = dish_name
                context.user_data['state'] = BotState.WAITING_FOR_INTENT.value
                sentiment = analyze_sentiment(replica)
                suffix = " В хорошем расположении духа? 😊" if sentiment == 'positive' else " Не грусти, найдем блюдо! 😊" if sentiment == 'negative' else ""
                return f"Из {dish_category} есть {dish_name}. Хотите узнать цену, состав или наличие?{suffix}"
        sentiment = analyze_sentiment(replica)
        suffix = " Отлично, давай продолжим! 😊" if sentiment == 'positive' else " Не переживай, уточним! 😊" if sentiment == 'negative' else ""
        return f"Пожалуйста, уточните название блюда или категорию.{suffix}"

    def _process_waiting_for_intent(self, replica, context):
        """Обрабатывает состояние WAITING_FOR_INTENT."""
        dish_name = extract_dish_name(replica)
        if dish_name and dish_name in CONFIG['dishes']:
            context.user_data['current_dish'] = dish_name
        else:
            dish_name = context.user_data.get('current_dish', 'блюдо')

        intent = self.classify_intent(replica)
        if intent in [Intent.DISH_PRICE.value, Intent.DISH_AVAILABILITY.value, Intent.DISH_INFO.value,
                      Intent.ORDER_DISH.value]:
            context.user_data['state'] = BotState.NONE.value
            return self._get_dish_response(intent, dish_name, replica, context)
        if intent == Intent.YES.value:
            if dish_name:
                context.user_data['state'] = BotState.NONE.value
                sentiment = analyze_sentiment(replica)
                suffix = " Рад твоему настроению! 😊" if sentiment == 'positive' else " Давай поднимем настроение! 😊" if sentiment == 'negative' else ""
                return f"Цена на {dish_name} — {CONFIG['dishes'][dish_name]['price']} рублей. Что ещё интересует?{suffix}"
        if intent == Intent.NO.value:
            context.user_data['current_dish'] = None
            context.user_data['state'] = BotState.NONE.value
            sentiment = analyze_sentiment(replica)
            suffix = " Отлично, продолжаем! 😊" if sentiment == 'positive' else " Не грусти, найдем другое! 😊" if sentiment == 'negative' else ""
            return f"Хорошо, какое блюдо обсудим теперь?{suffix}"
        sentiment = analyze_sentiment(replica)
        suffix = " В хорошем настроении? 😊" if sentiment == 'positive' else " Не переживай, найдем что-то вкусное! 😊" if sentiment == 'negative' else ""
        return f"Что хотите узнать про {dish_name}: цену, состав или наличие?{suffix}"

    def process(self, replica, context):
        """Обрабатывает запрос пользователя."""
        stats = Stats(context)
        if not is_meaningful_text(replica):
            answer = self.get_failure_phrase(replica)
            self._update_context(context, replica, answer)
            stats.add(ResponseType.FAILURE.value, replica, answer, context)
            return answer

        price = extract_price(replica)
        dish_category = extract_dish_category(replica)
        if price:
            answer = self._handle_filter_dishes(price, dish_category, context)
            self._update_context(context, replica, answer, Intent.FILTER_DISHES.value)
            stats.add(ResponseType.INTENT.value, replica, answer, context)
            return answer

        state = context.user_data.get('state', BotState.NONE.value)
        logger.info(
            f"Processing: replica='{replica}', state='{state}', last_intent='{context.user_data.get('last_intent')}'")

        if state == BotState.WAITING_FOR_DISH.value:
            answer = self._process_waiting_for_dish(replica, context)
        elif state == BotState.WAITING_FOR_INTENT.value:
            answer = self._process_waiting_for_intent(replica, context)
        else:
            answer = self._process_none_state(replica, context)

        self._update_context(context, replica, answer)
        stats.add(ResponseType.INTENT.value if self.classify_intent(
            replica) else ResponseType.GENERATE.value if 'dialogues.txt' in answer else ResponseType.FAILURE.value,
                  replica, answer, context)
        return answer

# Голос в текст
def voice_to_text(voice_file):
    recognizer = sr.Recognizer()
    try:
        import signal
        def signal_handler(signum, frame):
            raise TimeoutError("Speech recognition timed out")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(5)  # Таймаут 5 секунд
        audio = AudioSegment.from_ogg(voice_file)
        audio.export('voice.wav', format='wav')
        with sr.AudioFile('voice.wav') as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ru-RU')
        return text
    except (sr.UnknownValueError, sr.RequestError, TimeoutError, Exception) as e:
        logger.error(f"Ошибка распознавания голоса: {e}\n{traceback.format_exc()}")
        return None
    finally:
        signal.alarm(0)
        if os.path.exists('voice.wav'):
            os.remove('voice.wav')

# Текст в голос
def text_to_voice(text):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='ru')
        voice_file = 'response.mp3'
        tts.save(voice_file)
        return voice_file
    except Exception as e:
        logger.error(f"Ошибка синтеза речи: {e}\n{traceback.format_exc()}")
        return None

# Telegram-обработчики
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['start_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = Intent.HELLO.value
    await update.message.reply_text(answer)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = CONFIG['help_message']
    context.user_data['last_bot_response'] = answer
    context.user_data['last_intent'] = 'help'
    await update.message.reply_text(answer)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = context.user_data.get('stats', {ResponseType.INTENT.value: 0, ResponseType.GENERATE.value: 0,
                                            ResponseType.FAILURE.value: 0})
    answer = (
        f"Статистика:\n"
        f"Обработано намерений: {stats[ResponseType.INTENT.value]}\n"
        f"Ответов из диалогов: {stats[ResponseType.GENERATE.value]}\n"
        f"Неудачных запросов: {stats[ResponseType.FAILURE.value]}"
    )
    await update.message.reply_text(answer)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if not user_text:
        answer = "Пожалуйста, отправьте текст."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
        return
    bot = context.bot_data.setdefault('bot', Bot())
    answer = bot.process(user_text, context)
    await update.message.reply_text(answer)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    bot = context.bot_data.setdefault('bot', Bot())
    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive('voice.ogg')
        text = voice_to_text('voice.ogg')
        if text:
            answer = bot.process(text, context)
            voice_response = text_to_voice(answer)
            if voice_response:
                with open(voice_response, 'rb') as audio:
                    await update.message.reply_voice(audio)
                os.remove(voice_response)
            else:
                await update.message.reply_text(answer)
        else:
            answer = "Не удалось распознать голос. Попробуйте ещё раз."
            context.user_data['last_bot_response'] = answer
            await update.message.reply_text(answer)
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}\n{traceback.format_exc()}")
        answer = "Произошла ошибка. Попробуйте снова."
        context.user_data['last_bot_response'] = answer
        await update.message.reply_text(answer)
    finally:
        if os.path.exists('voice.ogg'):
            os.remove('voice.ogg')

def run_bot():
    if not TOKEN:
        raise ValueError("TELEGRAM_TOKEN не найден")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Бот запускается...")
    app.run_polling()

if __name__ == '__main__':
    run_bot()
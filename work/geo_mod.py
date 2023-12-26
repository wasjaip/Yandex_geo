
# Описание Библиотек и Их Назначение

# Обработка и Анализ Данных
import pandas as pd  # Для обработки и анализа данных (таблицы, временные ряды)
import numpy as np   # Для работы с многомерными массивами и математическими операциями
import torch
# Работа с Базами Данных
from sqlalchemy.engine.url import URL  # Формирование URL-адресов для подключения к БД
# Определение и взаимодействие с БД
from sqlalchemy import create_engine, MetaData, Table, Column, select,inspect, Integer, String, DECIMAL, CHAR, BIGINT, func, DATE, Float, Text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import ProgrammingError  # Обработка исключений программирования в SQLAlchemy
from sqlalchemy.inspection import inspect  # Инструменты для инспектирования объектов БД
from sqlalchemy.dialects.postgresql import insert  # Операция вставки для PostgreSQL

# Моделирование и Обработка Естественного Языка (NLP)
from sentence_transformers import SentenceTransformer, util, losses, evaluation, InputExample  # Для работы с моделями трансформеров в NLP
from sentence_transformers.util import cos_sim  # Вычисление косинусного сходства
from scipy.spatial.distance import pdist, squareform  # Расстояния между точками в многомерном пространстве

# Машинное Обучение и PyTorch
from sklearn.metrics.pairwise import cosine_similarity  # Вычисление косинусного сходства (scikit-learn)
from torch.utils.data import DataLoader  # Загрузка данных для моделей глубокого обучения

# Регулярные Выражения и Прочее
import re  # Работа с регулярными выражениями

from sqlalchemy import create_engine, inspect, text

import time
from sqlalchemy.exc import OperationalError




class geo():
    def __init__(self):
        self.engine = self.connect_db()
        self.metadata = MetaData(bind=self.engine)

    def connect_db(self):


        DATABASE = {
            'drivername': 'postgresql',
            'username': 'wasjaip',
            'password': 'qz4bdl63Q1977!',
            'host': '77.222.36.33',
            'port': '19679',
            'database': 'wasjaip'
        }

        # Создание строки подключения без пароля
        url_without_password = URL.create(
            drivername=DATABASE['drivername'],
            username=DATABASE['username'],
            password='',  # пароль опускается
            host=DATABASE['host'],
            port=DATABASE['port'],
            database=DATABASE['database']
        )

        # Создание engine с использованием connect_args для передачи пароля
        engine = create_engine(url_without_password, connect_args={"password": DATABASE['password']})

        return engine

    # функция создает таблицу geonames, если она не существует, и загружает в нее данные из файла geonames.txt.
    def create_and_load_geonames(self):
        # Названия столбцов для таблицы geonames
        column_names = [
            'geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude',
            'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code',
            'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation',
            'dem', 'timezone', 'modification_date'
        ]

        # Проверка наличия таблицы 'geonames' и создание, если она отсутствует
        if not inspect(self.engine).has_table('geonames'):
            # Создание таблицы 'geonames'
            geonames = Table('geonames', self.metadata,
                             Column('geonameid', Integer, primary_key=True),
                             Column('name', String(200)),
                             Column('asciiname', String(200)),
                             Column('alternatenames', Text),
                             Column('latitude', DECIMAL(9, 6)),
                             Column('longitude', DECIMAL(9, 6)),
                             Column('feature_class', CHAR(1)),
                             Column('feature_code', String(10)),
                             Column('country_code', CHAR(2)),
                             Column('cc2', String(60)),
                             Column('admin1_code', String(20)),
                             Column('admin2_code', String(80)),
                             Column('admin3_code', String(20)),
                             Column('admin4_code', String(20)),
                             Column('population', BIGINT),
                             Column('elevation', Integer),
                             Column('dem', Integer),
                             Column('timezone', String(40)),
                             Column('modification_date', DATE)
                             )
            self.metadata.create_all(self.engine)

            # Чтение данных из файла и загрузка в таблицу
            data = pd.read_csv('geonames.txt', sep='\t', names=column_names, dtype=str, encoding='utf-8')
            data.to_sql('geonames', con=self.engine, if_exists='append', index=False)
            print("Таблица 'geonames' создана и данные загружены.")
        else:
            print("Таблица 'geonames' уже существует.")

    def create_and_load_alternate_names(self):
        column_names_al = [
            'alternateNameId', 'geonameid', 'isolanguage', 'alternate_name', 'isPreferredName',
            'isShortName', 'isColloquial', 'isHistoric', 'from', 'to', 'link', 'wkdt'
        ]

        if not inspect(self.engine).has_table('alternateNames'):
            alternate_names = Table('alternateNames', self.metadata,
                                    Column('alternateNameId', Integer, primary_key=True),
                                    Column('geonameid', Integer),
                                    Column('isolanguage', String(7)),
                                    Column('alternate_name', String(400)),
                                    Column('isPreferredName', CHAR(1)),
                                    Column('isShortName', CHAR(1)),
                                    Column('isColloquial', CHAR(1)),
                                    Column('isHistoric', CHAR(1)),
                                    Column('from', Integer),
                                    Column('to', Integer),
                                    Column('link', String(255)),
                                    Column('wkdt', String(255)),
                                    extend_existing=True
                                    )
            self.metadata.create_all(self.engine)

            data = pd.read_csv('alternateNames.txt', sep='\t', encoding='utf-8', header=None)
            max_columns = 12
            data = data.reindex(columns=range(max_columns))
            data.columns = column_names_al

            data.to_sql('alternateNames', con=self.engine, if_exists='replace', index=False)
            print("Таблица 'alternateNames' создана и данные загружены.")
        else:
            print("Таблица 'alternateNames' уже существует.")

    def create_and_load_tables(self):
        self.create_and_load_geonames()
        self.create_and_load_alternate_names()


    # проверка наличия в базе
    def list_tables_and_columns(self):
        # Получение списка всех таблиц в базе данных
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        # Перебор таблиц и вывод информации о каждой
        for table_name in table_names:
            # Получение списка столбцов для каждой таблицы
            columns = inspector.get_columns(table_name)
            column_names = [column['name'] for column in columns]

            # Получение количества записей в каждой таблице
            query = f'SELECT COUNT(*) FROM "{table_name}"'
            record_count = self.engine.execute(query).scalar()

            # Вывод информации
            print(f"Таблица: {table_name}")
            print(f"Столбцы: {column_names}")
            print(f"Количество записей: {record_count}")
            print("\n")  # Для разделения информации о разных таблицах

    def read_and_combine_tables(self):
        # Определение запроса для основных таблиц
        print('загружаю базы')
        query_main = """
        SELECT geonameid, name, asciiname, country_code, admin1_code, admin2_code
        FROM {}
        WHERE country_code IN ('RU', 'BY', 'KG', 'KZ', 'AM', 'GE', 'RS', 'ME')
        """

        # Запрос для таблицы alternateNames
        query_alt_names = """
        SELECT geonameid, alternate_name, "isPreferredName", "isShortName", "isColloquial", "isHistoric", "to"
        FROM "alternateNames"
        """

        # Чтение данных из основных таблиц
        df_main = pd.concat([
            pd.read_sql_query(query_main.format(f'"{table}"'), self.engine)
            for table in ['allCountries', 'cities', 'cities15000', 'cities500', 'cities5000', 'geonames']
        ], ignore_index=True)

        # Чтение данных из таблицы с альтернативными названиями
        df_alt_names = pd.read_sql_query(query_alt_names, self.engine)

        # Объединение данных
        df_combined = pd.merge(df_main, df_alt_names, on='geonameid', how='left')
        print('базы готовы')
        return df_combined

    @staticmethod
    # Вспомогательные функции
    def contains_only_nan(lst):
        return all(pd.isna(item) for item in lst)

    @staticmethod
    def replace_empty_with_asciiname(alternate_name, asciiname):
        if all(pd.isna(name) or name == '' for name in alternate_name):
             return [asciiname]
        return alternate_name

    def process_and_analyze_data(self, df_combined):

        """
        Обрабатывает и анализирует данные, производя ряд преобразований и агрегаций.
        """
        print('подговаливаем для эмбейдингов')
        #df_combined = self.read_and_combine_tables()

        # Преобразование всех значений в столбце alternate_name в строки
        df_combined['alternate_name'] = df_combined['alternate_name'].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))

        # Подсчет количества значений в столбце alternate_name, содержащих 'https://'
        count_links = df_combined['alternate_name'].str.contains('https://').sum()
        print(f"Количество значений со ссылками 'https://': {count_links}")

        # Удаление значений в столбце alternate_name, содержащих 'https://'
        df_combined['alternate_name'] = df_combined['alternate_name'].apply(
            lambda x: "" if 'https://' in x else x)

        # Фильтрация ненужных столбцов
        filtered_df = df_combined[['asciiname', 'country_code', 'admin1_code', 'alternate_name']]

        # Группировка данных и агрегация всех alternate_name
        grouped_df = filtered_df.groupby(['asciiname', 'country_code', 'admin1_code'])['alternate_name'].apply(
            list).reset_index()

        # Удаление дубликатов в списках столбца alternate_name
        grouped_df['alternate_name'] = grouped_df['alternate_name'].apply(lambda x: list(set(x)))

        # Применение функций замены для списков с NaN или пустыми строками

        grouped_df['alternate_name'] = grouped_df.apply(
            lambda row: self.replace_empty_with_asciiname(row['alternate_name'], row['asciiname']),
            axis=1
        )

        # Подсчет количества каждого уникального значения в столбце 'country_code'
        country_code_counts = grouped_df['country_code'].value_counts()
        print(country_code_counts)

        return grouped_df


    def fine_tune_model(self, model_name, loss_function, dataframe=None, epochs=8, warmup_steps=100, evaluation_steps=10000):
        """
        Дообучает модель SentenceTransformer на предоставленных данных.

        :param model_name: Имя модели для дообучения.
        :param loss_function: Выбор функции потерь.
        :param dataframe: DataFrame с данными для обучения. Если None, используется результат process_and_analyze_data.
        :param epochs: Количество эпох обучения.
        :param warmup_steps: Количество шагов разогрева.
        :param evaluation_steps: Шаги для оценки во время обучения.
        """
        # Загрузка предобученной модели
        model = SentenceTransformer(model_name)

        # Если DataFrame не предоставлен, используем результат process_and_analyze_data
        if dataframe is None:
            dataframe = self.process_and_analyze_data(self.read_and_combine_tables())

        # Подготовка данных
        dataframe = dataframe.explode('alternate_name')
        dataframe = dataframe[dataframe.asciiname != dataframe.alternate_name]
        dataframe = dataframe.drop_duplicates(subset=['asciiname', 'alternate_name'])
        dataframe = dataframe.dropna(subset=['alternate_name'])
        dataframe = dataframe[dataframe['alternate_name'].str.strip().str.len() > 0]

        # Создание InputExample объектов
        dataframe['example'] = dataframe.apply(lambda x: InputExample(texts=[x['asciiname'], x['alternate_name']]), axis=1)

        # Проверка на наличие пустых текстов
        for example in dataframe['example']:
            if not all(example.texts):
                raise ValueError("Обнаружен InputExample с пустым текстом")

        # Создание DataLoader
        train_data = dataframe['example'].tolist()
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)

        # Выбор функции потерь
        if loss_function == 'cosine':
            train_loss = losses.CosineSimilarityLoss(model)
        elif loss_function == 'ranking':
            train_loss = losses.MultipleNegativesRankingLoss(model)
        else:
            raise ValueError("Неверно указана функция потерь")

        # Определение оценщика и параметров обучения
        print('обучение началось')
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(train_data, name='asciiname-evaluator')
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps, evaluator=evaluator, evaluation_steps=evaluation_steps)

        return model
    def prepare_embeddings(self, model_name, dataframe, column_name):
        """
        Подготавливает эмбеддинги для указанной колонки из DataFrame.

        :param model_name: Имя модели для генерации эмбеддингов.
        :param dataframe: DataFrame с данными.
        :param column_name: Название колонки в DataFrame для генерации эмбеддингов.
        """
        self.model = SentenceTransformer(model_name)
        unique_names = dataframe[column_name].drop_duplicates().values
        self.embeddings = self.model.encode(unique_names, show_progress_bar=True)
        self.names = unique_names

    def get_similar_names(self, geo_name, top=1):
        """
        Ищет наиболее похожие названия на основе заданного названия.

        :param geo_name: Название для поиска похожих.
        :param top: Количество наиболее похожих названий для возврата.
        :return: DataFrame с результатами поиска.
        """
        query_embedding = self.model.encode([geo_name], show_progress_bar=False)
        search_results = util.semantic_search(query_embedding, self.embeddings, top_k=top)[0]
        search_results = search_results[:top]
        result_df = pd.DataFrame(search_results)
        result_df = result_df.assign(name=[self.names[i] for i in result_df['corpus_id']])

        return result_df

    def test_model(self, file_path):
        """
        Тестирует модель на данных из файла и вычисляет точность.

        :param file_path: Путь к файлу с тестовыми данными.
        :param model: Обученная модель для тестирования.
        """

        if not hasattr(self, 'model'):
            raise Exception("Модель не загружена")


        # Попытка чтения файла с разными кодировками
        try:
            data_test = pd.read_csv(file_path, encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            try:
                data_test = pd.read_csv(file_path, encoding='ISO-8859-1', sep=';')
            except UnicodeDecodeError:
                data_test = pd.read_csv(file_path, encoding='cp1251', sep=';')

        # Применение функции get_all к каждому элементу в столбце query

        data_test['result'] = data_test['query'].apply(lambda x: self.get_similar_name(x, top=1))

        # Расчет точности
        accuracy = (data_test['result'] == data_test['name']).mean()
        print(f"Точность: {accuracy * 100:.2f}%")

    def get_similar_name(self, geo_name, model, top=1):
        """
        Возвращает наиболее похожее название на основе заданного названия.

        :param geo_name: Название для поиска похожих.
        :param model: Обученная модель для поиска.
        :param top: Количество наиболее похожих названий для возврата.
        """
        if not hasattr(self, 'model'):
            raise Exception("Модель не загружена")

        query_embedding = self.model.encode([geo_name], show_progress_bar=False)
        search_results = util.semantic_search(query_embedding, self.embeddings, top_k=top)[0]
        return self.names[search_results[0]['corpus_id']]

    def load_embeddings(self, file_path):
        """
        Загружает эмбеддинги из файла.

        :param file_path: Путь к файлу с эмбеддингами.
        """
        with open(file_path, 'rb') as f:
            self.embeddings = pickle.load(f)

    def load_embeddings_from_db(self):
        """
        Загружает эмбеддинги из таблицы базы данных.
        """
        if self.engine is None:
            raise Exception("Сначала необходимо подключиться к базе данных")
        start_time = time.time()  # Начало отсчета времени

        # Выполнение запроса для получения эмбеддингов
        with self.engine.connect() as conn:
            result = conn.execute("SELECT * FROM emb")
            embeddings_data = result.fetchall()


        # Преобразование данных в NumPy массив с типом float32
        self.embeddings = np.array([np.array(row[1:], dtype=np.float32) for row in embeddings_data])

        end_time = time.time()  # Конец отсчета времени
        print(f"Загрузка эмбеддингов завершена за {end_time - start_time:.2f} секунд")

    def test_model1(self, file_path):
        """
        Тестирует модель на данных из файла и вычисляет точность.

        :param file_path: Путь к файлу с тестовыми данными.
        """
        if self.model is None or self.embeddings is None:
            raise Exception("Модель или эмбеддинги не загружены")

        # Чтение тестовых данных
        try:
            data_test = pd.read_csv(file_path, encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            try:
                data_test = pd.read_csv(file_path, encoding='ISO-8859-1', sep=';')
            except UnicodeDecodeError:
                data_test = pd.read_csv(file_path, encoding='cp1251', sep=';')

        # Применение функции get_similar_name к каждому элементу в столбце query
        data_test['result'] = data_test['query'].apply(lambda x: self.get_similar_name(x, top=1))

        # Расчет точности
        accuracy = (data_test['result'] == data_test['name']).mean()
        print(f"Точность: {accuracy * 100:.2f}%")

    def get_similar_name(self, geo_name, top=1):
        """
        Возвращает наиболее похожее название на основе заданного названия.

        :param geo_name: Название для поиска похожих.
        :param top: Количество наиболее похожих названий для возврата.
        """
        query_embedding = self.model.encode([geo_name], show_progress_bar=False)
        search_results = util.semantic_search(query_embedding, self.embeddings, top_k=top)[0]
        return self.names[search_results[0]['corpus_id']]

    def load_model(self, model_name):
        """
        Загружает модель SentenceTransformer и сохраняет в атрибут класса.

        :param model_name: Имя модели для загрузки.
        """
        self.model = SentenceTransformer(model_name)


    def table_exists(self, table_name):
        """
        Проверяет наличие таблицы в базе данных.

        :param table_name: Название таблицы для проверки.
        :return: Возвращает True, если таблица существует, иначе False.
        """
        if self.engine is None:
            raise Exception("Сначала необходимо подключиться к базе данных")

        inspector = Inspector.from_engine(self.engine)
        return table_name in inspector.get_table_names()


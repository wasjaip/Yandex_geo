# Описание проекта

## Клиент
Карьерный центр Яндекс

## Цель проекта
Разработка инструмента для сопоставления произвольных географических названий с унифицированными именами из базы данных geonames для внутреннего использования Карьерным центром.

## Основные задачи
1. Создание решения для подбора наиболее подходящих названий (преобразование локальных названий в международные эквиваленты, например, Ереван -> Yerevan).
2. Фокус на данных о городах из России и стран, популярных для релокации (Беларусь, Армения, Казахстан, Кыргызстан, Турция, Сербия), с населением от 15000 человек.
3. Формат выходных данных: список словарей с полями geonameid, name, region, country и cosine similarity.

## Дополнительные задачи (опционально)
- Настройка количества предлагаемых названий.
- Коррекция ошибок и опечаток в названиях.
- Хранение данных geonames и векторизованных данных в PostgreSQL.
- Методы для настройки подключения к базе данных и инициализации класса.
- Методы для добавления новых географических названий.

## Результат проекта
- [База PostgreSQL 15, развернута online на севрвере 2 CPU, 2 ГБ RAM, 40 ГБ NVMe](https://vps.sweb.ru/dbaas)
- [Модель на huggingface, обучена на 8 эпохах и полном датафрейме всех наименований](https://huggingface.co/wasjaip/LaBSE_geonames_v1)
- [Архив эмбедингов для общего датафрейма](https://github.com/wasjaip/Yandex_geo/blob/main/work/emb.rar)
- [Класс](https://github.com/wasjaip/Yandex_geo/blob/main/work/geo_mod.py)
- [Тестовый датасет](https://github.com/wasjaip/Yandex_geo/blob/main/work/geo_test.csv)
- [Тетрадка для дополнительного обучения и проверки эффективности](https://github.com/wasjaip/Yandex_geo/blob/main/work/test_psgl.ipynb)
- [Тетрадка для встройки](https://github.com/wasjaip/Yandex_geo/blob/main/work/user_psgl.ipynb)


## Используемые данные
- Таблицы с geonames (admin1CodesASCII, alternateNamesV2, cities15000, countryInfo и другие).
- Тестовый датасет с Yandex Disk.

## Используемый технологический стек
- Библиотеки для ML: SQL, Pandas, NLP, Transformers.
- Дополнительные инструменты: SQLAlchemy, CountVectorizer, TfIdfVectorizer, Sentence transformers, pyspellchecker, translate, transliterate, spaCy, Hugging Face.

## Сроки
Предварительный срок реализации проекта – 3 недели

## План реализации
1. Изучение базы данных Geonames.
2. Тестирование и выбор моделей.
3. Оформление, упаковка проекта, настройка модуля для интеграции.


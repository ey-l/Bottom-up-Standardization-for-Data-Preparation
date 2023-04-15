
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from geopy.geocoders import Nominatim
from pyspark.sql.window import Window
import pandas as pd
import missingno
import copy
import re
from nltk.corpus import stopwords
import nltk
import plotly.express as px
import plotly.graph_objects as go
INPUTFOLDER = '_data/input/competitive-data-science-predict-future-sales/'
spark = SparkSession.builder.config('spark.driver.memory', '15g').getOrCreate()
read_spark = lambda p: spark.read.csv(f'{INPUTFOLDER}{p}.csv', inferSchema=True, header=True)
hardcode_replacer = {'360, ан', 'с"", иг', ' (7,5 г', 'я"", ар', 'и"", Кр', 'PS3, ру', 'Рид, То', '8"", се', 'м"", ар', 'а"", De', '[PC, Ци', 'в"", ар', 'k)", ар', 'м"", ию', 'а"", ар', 'PS4, ан', '"" , ар', 'с"",Кри', 'ень, ро', 'алл, Дэ', 'тер, Дж', 'ижу, ни', 'MP3, Го', 'ени, 2 ', '""9,8""', 'и"", кр', ' PC, Ци', '[PC, ру', 'а"", ма', 'и"", 3 ', 'уин, Хо', 'шт., ро', 'm"", ар', 'вом, ра', 'й"", 9*', 'аук, Но', 'й"", А.', 'MP3, ИД', '(11,5 г', 'e)",арт', '"Ну, по', 'y)", ар', 'и"", ар', 'С"", N ', '- 4, 5 ', 'КИ), 25', 'я"", Ма', 'ртс, мя', ')"", ар', 'изд, Ба', 'ски, Ба', 'г"", ар', '8"", ""', 'орд, Га', 'ах", 7 ', ' 11,5 г', 'и"", же', 'и"", че', 'зки, бе', 'een, Re', 'ах", 7 ', 'PS4, ру', 'ора, Ло', '[PC, Je', 'тка, 3 ', 'Н.Ю, Ба', 'кин, Ф.', 'old, Gr', '9см, се', 'ack, Re', 'ова, Ло', 'йсе, ар', 'нто, Ло', 'ска, бе', ' S4, иг', 'юкс, Ло', 'онс, Си', 'сса, Ми', 'рик, ла', 'sen, MP', 'сси, Ло', 'e)",арт', 'N 8, ав', 'лки, 2 ', 'ах", 7 ', 'рый, ар', 'wel, ру', 'e)",арт', 'ах", 7 ', 'e)",арт', 's Blue)"",арт.', 'Black)"", а', 'y)"", ар', 'en)"", ар', 'тах"", 7', 'ветах, 3 и'}
with open(f'{INPUTFOLDER}items.csv', 'r') as file:
    a = file.read()
    for r in hardcode_replacer:
        a = a.replace(r, r.replace(',', '.'))
with open(f'./items.csv', 'w+') as file:
    file.write(a)
items = spark.read.csv(f'./items.csv', inferSchema=True, header=True)
shops = read_spark('shops')
cat = read_spark('item_categories')
train = read_spark('sales_train')
test = read_spark('test')
sample = read_spark('sample_submission')
print(f'Number of records in *shops* table : {shops.count()}')
shops.show(5, truncate=False)
print(f'Number of records in *cat* table : {cat.count()}')
cat.show(5, truncate=False)
print(f'Number of records in *items* table : {items.count()}')
items.show(5, truncate=False)
print(f'Number of records in *train* table : {train.count()}')
train.show(5, truncate=False)
print(f'Number of records in *test* table : {test.count()}')
test.show(5, truncate=False)
train = train.withColumn('date', f.to_timestamp(train.date, 'dd.MM.yyyy'))
train = train.withColumn('year', f.year(train.date))
train = train.withColumn('month', f.month(train.date))
train.show()
print(f'Number of records in the train set before the monthly grouping: {train.count()}')
col = ['date_block_num', 'year', 'month', 'shop_id', 'item_id']
train = train.groupby(col).sum('item_cnt_day').select(col + [f.col('sum(item_cnt_day)').alias('item_cnt_month')])
print(f'Number of records in train set after the monthly grouping: {train.count()}')
train.show()
print(f"Number of unique items in Train dataset: {train.select('item_id').distinct().count()}")
items_for_pred = [i[0] for i in test.select('item_id').distinct().collect()]
print(f'Number of unique items in Test dataset: {len(items_for_pred)}')
print(f'Filtering the train set & items set with such items ...')
print(f'Length of train set [{train.count()}] rec. -----> ', end='')
train = train.filter(train.item_id.isin(items_for_pred))
print(f'[{train.count()}] rec.')
print(f'Length of items set [{items.count()}] rec. -----> ', end='')
items = items.filter(items.item_id.isin(items_for_pred))
print(f'[{items.count()}] rec.')
print(f"Number of unique shops in Train dataset: {train.select('shop_id').distinct().count()}")
shops_for_pred = [i[0] for i in test.select('shop_id').distinct().collect()]
print(f'Number of unique shops in Test dataset: {len(shops_for_pred)}')
print(f'Filtering the train set & shops set with such items ...')
print(f'Length of train set [{train.count()}] rec. -----> ', end='')
train = train.filter(train.shop_id.isin(shops_for_pred))
print(f'[{train.count()}] rec.')
print(f'Length of items set [{shops.count()}] rec. -----> ', end='')
shops = shops.filter(shops.shop_id.isin(shops_for_pred))
print(f'[{shops.count()}] rec.')
_shops = test.select('shop_id').distinct()
_items = test.select('item_id').distinct()
_blocks = train.select(['date_block_num', 'year', 'month']).distinct()
allcomb = _shops.crossJoin(_items).crossJoin(_blocks)
print(f'The len of the dataset should be equel to {allcomb.count()}')
print('to consider all combinations of shop_id & item_id for each month')
train = allcomb.join(train.alias('t'), (allcomb.item_id == f.col('t.item_id')) & (allcomb.shop_id == f.col('t.shop_id')) & (allcomb.date_block_num == f.col('t.date_block_num')), 'left').select([allcomb.item_id, allcomb.shop_id, allcomb.date_block_num, allcomb.month, allcomb.year, f.col('t.item_cnt_month')])
train = train.na.fill({'item_cnt_month': 0})
(N, Y, M) = (34, 2015, 11)
test = test.withColumn('date_block_num', f.lit(N)).withColumn('year', f.lit(Y)).withColumn('month', f.lit(M)).withColumn('item_cnt_month', f.lit(None)).drop('ID')
train = train.union(test.select(train.columns))
a = train.groupby(['year', 'month']).agg({'item_cnt_month': 'sum'}).toPandas()
px.line(a.sort_values('month'), x='month', y='sum(item_cnt_month)', color='year')
windowSpec = Window().partitionBy(['item_id', 'shop_id']).orderBy(['date_block_num'])
for i in range(10):
    train = train.withColumn(f'lag_{i + 1}', f.lag('item_cnt_month', i + 1).over(windowSpec))
train.filter(~f.isnan(f.col('lag_10'))).show()
new_cat_labels = {'PC - Гарнитуры/Наушники': 'analog  - Аксессуары - PC         - Гарнитуры/Наушники -', 'Аксессуары - PS2': 'analog  - Аксессуары - PS         - PS2                -', 'Аксессуары - PS3': 'analog  - Аксессуары - PS         - PS3                -', 'Аксессуары - PS4': 'analog  - Аксессуары - PS         - PS4                -', 'Аксессуары - PSP': 'analog  - Аксессуары - PS         - PSP                -', 'Аксессуары - PSVita': 'analog  - Аксессуары - PS         - PSVita             -', 'Аксессуары - XBOX 360': 'analog  - Аксессуары - XBOX       - 360                -', 'Аксессуары - XBOX ONE': 'analog  - Аксессуары - XBOX       - ONE                -', 'Игровые консоли - PS2': 'analog  - Консоли    - PS         - PS2                -', 'Игровые консоли - PS3': 'analog  - Консоли    - PS         - PS2                -', 'Игровые консоли - PS4': 'analog  - Консоли    - PS         - PS3                -', 'Игровые консоли - PSP': 'analog  - Консоли    - PS         - PSP                -', 'Игровые консоли - PSVita': 'analog  - Консоли    - PS         - PSVita             -', 'Игровые консоли - XBOX 360': 'analog  - Консоли    - XBOX       - 360                -', 'Игровые консоли - XBOX ONE': 'analog  - Консоли    - XBOX       - ONE                -', 'Игровые консоли - Прочие': 'analog  - Консоли    - Консоли    - Консоли            -', 'Игры - PS2': 'analog  - Игры       - PS         - PS2                -', 'Игры - PS3': 'analog  - Игры       - PS         - PS3                -', 'Игры - PS4': 'analog  - Игры       - PS         - PS4                -', 'Игры - PSP': 'analog  - Игры       - PS         - PSP                -', 'Игры - PSVita': 'analog  - Игры       - PS         - PSVita             -', 'Игры - XBOX 360': 'analog  - Игры       - XBOX       - 360                -', 'Игры - XBOX ONE': 'analog  - Игры       - XBOX       - ONE                -', 'Игры - Аксессуары для игр': 'analog  - Игры       - Аксессуары - Для игр            -', 'Игры Android - Цифра': 'digital - Игры       - Android    - Android            -', 'Игры MAC - Цифра': 'digital - Игры       - MAC        - MAC                -', 'Игры PC - Дополнительные издания': 'analog  - Игры       - PC         - Дополнительные     -', 'Игры PC - Коллекционные издания': 'analog  - Игры       - PC         - Коллекционные      -', 'Игры PC - Стандартные издания': 'analog  - Игры       - PC         - Стандартные        -', 'Игры PC - Цифра': 'digital - Игры       - PC         - Стандартные        -', 'Карты оплаты (Кино, Музыка, Игры)': 'analog  - Карты      - Карты      - Кино/Музыка/Игры   -', 'Карты оплаты - Live!': 'analog  - Карты      - XBOX       - Live!              -', 'Карты оплаты - Live! (Цифра)': 'digital - Карты      - XBOX       - Live!              -', 'Карты оплаты - PSN': 'analog  - Карты      - PS         - PSN                -', 'Карты оплаты - Windows (Цифра)': 'digita  - Карты      - PC         - Windows            -', 'Кино - Blu-Ray': 'analog  - Кино       - Blu_Ray    - Стандартные        -', 'Кино - Blu-Ray 3D': 'analog  - Кино       - Blu_Ray    - 3D                 -', 'Кино - Blu-Ray 4K': 'analog  - Кино       - Blu_Ray    - 4K                 -', 'Кино - DVD': 'analog  - Кино       - DVD        - Стандартные        -', 'Кино - Коллекционное': 'analog  - Кино       - Кино       - Коллекционные      -', 'Книги - Артбуки, энциклопедии': 'analog  - Книги      - Арт        - Артбуки            -', 'Книги - Аудиокниги': 'analog  - Книги      - Аудиокниги - Аудиокниги         -', 'Книги - Аудиокниги (Цифра)': 'digital - Книги      - Аудиокниги - Аудиокниги         -', 'Книги - Аудиокниги 1С': 'analog  - Книги      - Аудиокниги - 1С                 -', 'Книги - Бизнес литература': 'analog  - Книги      - Развитие   - Бизнес             -', 'Книги - Комиксы, манга': 'analog  - Книги      - Арт        - Комиксы/Манга      -', 'Книги - Компьютерная литература': 'analog  - Книги      - Развитие   - Компьютерная       -', 'Книги - Методические материалы 1С': 'analog  - Книги      - Развитие   - 1С                 -', 'Книги - Открытки': 'analog  - Книги      - Открытки   - Открытки           -', 'Книги - Познавательная литература': 'analog  - Книги      - Развитие   - Познавательная     -', 'Книги - Путеводители': 'analog  - Книги      - Книги      - Путеводители       -', 'Книги - Художественная литература': 'analog  - Книги      - Книги      - Художественная     -', 'Книги - Цифра': 'digital - Книги      - Книги      - Книги              -', 'Музыка - CD локального производства': 'analog  - Музыка     - CD         - Локальные          -', 'Музыка - CD фирменного производства': 'analog  - Музыка     - CD         - Фирменные          -', 'Музыка - MP3': 'analog  - Музыка     - MP3        - MP3                -', 'Музыка - Винил': 'analog  - Музыка     - Винил      - Винил              -', 'Музыка - Музыкальное видео': 'analog  - Музыка     - Видео      - Видео              -', 'Музыка - Подарочные издания': 'analog  - Музыка     - Музыка     - Подарочные         -', 'Подарки - Атрибутика': 'analog  - Подарки    - Подарки    - Атрибутика         -', 'Подарки - Гаджеты, роботы, спорт': 'analog  - Подарки    - Подарки    - Гаджеты            -', 'Подарки - Мягкие игрушки': 'analog  - Подарки    - Игрушки    - Мягкие             -', 'Подарки - Настольные игры': 'analog  - Подарки    - Настольные - Обычные            -', 'Подарки - Настольные игры (компактные)': 'analog  - Подарки    - Настольные - Компактные         -', 'Подарки - Открытки, наклейки': 'analog  - Подарки    - Открытки   - Открытки/Наклейки  -', 'Подарки - Развитие': 'analog  - Подарки    - Развитие   - Развитие           -', 'Подарки - Сертификаты, услуги': 'analog  - Подарки    - Услуги     - Сертификаты        -', 'Подарки - Сувениры': 'analog  - Подарки    - Сувениры   - Сувениры           -', 'Подарки - Сувениры (в навеску)': 'analog  - Подарки    - Сувениры   - В навеску          -', 'Подарки - Сумки, Альбомы, Коврики д/мыши': 'analog  - Подарки    - Аксессуары - Альбомы/Коврики    -', 'Подарки - Фигурки': 'analog  - Подарки    - Игрушки    - Фигурки            -', 'Программы - 1С:Предприятие 8': 'analog  - Программы  - Программы  - 1С                 -', 'Программы - MAC (Цифра)': 'digital - Программы  - MAC        - MAC                -', 'Программы - Для дома и офиса': 'analog  - Программы  - Программы  - Для дома и офиса   -', 'Программы - Для дома и офиса (Цифра)': 'digital - Программы  - Программы  - Для дома и офиса   -', 'Программы - Обучающие': 'analog  - Программы  - Программы  - Обучающие          -', 'Программы - Обучающие (Цифра)': 'digital - Программы  - Программы  - Обучающие          -', 'Служебные': 'analog  - Служебные  - Служебные  - Служебные          -', 'Служебные - Билеты': 'analog  - Служебные  - Служебные  - Билеты             -', 'Чистые носители (шпиль)': 'analog  - Носители   - Носители   - Шпиль              -', 'Чистые носители (штучные)': 'analog  - Носители   - Носители   - Шт                 -', 'Элементы питания': 'analog  - Эл.питания - Эл.питания - Элементы питания   -', 'Билеты': 'analog  - Билеты     - Билеты     - Билеты             -', 'Билеты (Цифра)': 'digital - Билеты     - Билеты     - Билеты             -', 'Доставка товара': 'analog  - Доставка   - Услуги     - Доставка           -'}
cat = cat.na.replace(new_cat_labels)
split_col = f.split(cat.item_category_name, '-')
cat = cat.withColumn('is_digital', f.trim(split_col.getItem(0))).withColumn('class', f.trim(split_col.getItem(1))).withColumn('category', f.trim(split_col.getItem(2))).withColumn('sub-category', f.trim(split_col.getItem(3)))
cat = cat.select(['item_category_id', 'is_digital', 'class', 'category', 'sub-category'])
cat.show(84)
print(items.count())
items.show(truncate=False)

def clean_text(txt):
    txt = re.sub('[^+A-Za-zА-Яа-я0-9]+', ' ', str(txt).lower()).strip()
    txt = ' '.join([word for word in txt.split(' ') if len(word) > 1])
    return txt
spark_clean_text = f.UserDefinedFunction(clean_text)
items = items.withColumn('item_name', spark_clean_text(items['item_name']))
items.show(truncate=False)

def drop_stopwords(txt):
    from nltk.corpus import stopwords
    txt = ' '.join([s for s in txt.split() if s not in stopwords.words('english') and s not in stopwords.words('russian')])
    return txt
spark_drop_stopwords = f.UserDefinedFunction(drop_stopwords)
items = items.withColumn('item_name', spark_drop_stopwords(items['item_name']))
items.show(truncate=False)
items = items.withColumn('item_name', f.split(f.col('item_name'), ' '))
words_freq = items.withColumn('word', f.explode('item_name')).groupBy('word').count()
extra_cat = words_freq.filter(words_freq['count'] > items.count() / 100).select('word')
extra_cat = [i[0] for i in extra_cat.collect()]
print(f'number of extra features to add: {len(extra_cat)}')
for k in extra_cat:
    spark_get_extra_cat = f.UserDefinedFunction(lambda x: 1 if k in x else 0)
    items = items.withColumn(k, spark_get_extra_cat(items['item_name']).cast(pyspark.sql.types.IntegerType()))
items = items.select(['item_id', 'item_category_id'] + extra_cat)
print(items.columns)
shops.show(truncate=False)
shops = shops.withColumn('shop_name', spark_clean_text(shops['shop_name']))
shops = shops.withColumn('shop_name', spark_drop_stopwords(shops['shop_name']))
shops.show(truncate=False)

def get_city(shop_name):
    return shop_name.split(' ')[0]
spark_get_city = f.UserDefinedFunction(get_city)
shops = shops.withColumn('city', spark_get_city(shops['shop_name']))
print(shops.select('city').distinct().count())
shops.show(truncate=False)
geolocator = Nominatim(user_agent='Your_Name')
city_to_latlon = {}
for city in shops.select('city').distinct().collect():
    city = city[0]
    c = city if city != 'ростовнадону' else 'ростов-на-дону'
    if c not in ['цифровой', 'интернет', 'выездная']:
        location = geolocator.geocode(c)
        city_to_latlon[city] = (location.latitude, location.longitude)
    else:
        city_to_latlon[city] = (0, 0)
spark_get_latlon = f.UserDefinedFunction(lambda k: city_to_latlon[k][0])
shops = shops.withColumn('lat', spark_get_latlon(shops['city']))
spark_get_latlon = f.UserDefinedFunction(lambda k: city_to_latlon[k][1])
shops = shops.withColumn('lon', spark_get_latlon(shops['city']))
shops.show(truncate=False)

def get_type(shop_name):
    for i in ['трк', 'тц', 'трц', 'тк']:
        if i in shop_name:
            return i
    return 'no_shop_type'
spark_get_type = f.UserDefinedFunction(get_type)
shops = shops.withColumn('shop_type', spark_get_type(shops['shop_name']))
shops.show(truncate=False)
shops = shops.select(['shop_id', 'lat', 'lon', 'shop_type'])
shops.show()
dataset = items.join(cat, cat.item_category_id == items.item_category_id, 'left').select([items.item_category_id, 'item_id', 'is_digital', 'class', 'category', 'sub-category'] + extra_cat)
dataset = train.join(dataset, dataset.item_id == train.item_id, 'left').select([train.item_id, 'sub-category', 'date_block_num', 'item_cnt_month', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'is_digital', 'class', 'category', 'shop_id', 'month', 'year'] + extra_cat)
dataset = dataset.join(shops, shops.shop_id == dataset.shop_id, 'left').select([dataset.shop_id, 'item_id', 'date_block_num', 'item_cnt_month', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'is_digital', 'class', 'category', 'sub-category', 'lat', 'lon', 'shop_type', 'month', 'year'] + extra_cat)
train.filter(f.isnull(f.col('item_cnt_month'))).count()
dataset.repartition(10).write.csv('dataset', sep='|')
print(dataset.columns)
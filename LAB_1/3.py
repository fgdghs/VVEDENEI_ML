import pandas as pd

messages = [
    "Привет всем участникам чата",
    "Розыгрыш iPhone прямо сейчас!!!",
    "Кто знает хороший курс по Python?",
    "@admin помогите разобраться",
    "КРИПТА только растет #bitcoin $$$",
    "Отличная статья про машинное обучение",
    "Розыгрыш* призов каждый день",
    "Подскажите книгу для начинающих",
    "Заработок на крипте легко",
    "Спасибо за помощь",
    "### Супер предложение ###",
    "Изучаю pandas для анализа данных",
    "Розыгрыш 1000$ для всех",
    "Какой фреймворк выбрать для веба?",
    "Крипта это будущее финансов",
]

df = pd.DataFrame({'message': messages})


# 1
df['is_spam'] = df['message'].str.contains('крипта|розыгрыш|крипте',case=False)# case - для учёта регистра,

# 2
df['message'] = df['message'].str.replace(r'[@#$*]', '', regex=True)

# 3
df['word_count'] = df['message'].apply(lambda x: len(x.split())) #apply - функция применятеся к каждому элементу серии

# 4
result_df = df[(df['word_count'] > 3) & (df['is_spam'] == False)]

print(result_df[['message', 'word_count', 'is_spam']])
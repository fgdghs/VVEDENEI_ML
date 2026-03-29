import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


df = pd.read_csv('LAB_2/final_depression_dataset_1.csv')

print(df.isnull().sum())

# 1. ПЕРВИЧНАЯ ОЧИСТКА
if 'Name' in df.columns:
    df = df.drop(columns=['Name'])

df = df.drop_duplicates()

print(df.isnull().sum())
print("\n")

# 2. Обработкаы
df['is_student'] = df['Working Professional or Student'].apply(lambda x: 1 if x == 'Student' else 0)


df.loc[df['is_student'] == 1, ['Work Pressure', 'Job Satisfaction']] = 0
df.loc[df['is_student'] == 0, ['Academic Pressure', 'Study Satisfaction']] = 0

print(df.isnull().sum())
print("\n")

if 'CGPA' in df.columns and 'Degree' in df.columns:
    df['CGPA'] = df['CGPA'].fillna(df.groupby('Degree')['CGPA'].transform('median'))

chislovye = df.select_dtypes(include=[np.number]).columns
df[chislovye] = df[chislovye].fillna(df[chislovye].median())

kategorialnye = df.select_dtypes(include=['object']).columns
for col in kategorialnye:
    df[col] = df[col].fillna('Unknown')

print(df.isnull().sum())
print("\n")

# 4. КОДИРОВАНИЕ
kodirovshik = LabelEncoder()
for stolbec in df.columns:
    if not pd.api.types.is_numeric_dtype(df[stolbec]):
        df[stolbec] = kodirovshik.fit_transform(df[stolbec].astype(str))


# 5. ОБУЧЕНИЕ МОДЕЛИ
celevaya_peremennaya = 'Depression' if 'Depression' in df.columns else 'Depressed'
X = df.drop(columns=[celevaya_peremennaya])
y = df[celevaya_peremennaya]

X_obuchenie, X_test, y_obuchenie, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_lesa = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
model_lesa.fit(X_obuchenie, y_obuchenie)

# 6. ВИЗУАЛИЗАЦИЯ
prognoz = model_lesa.predict(X_test)
print(f"Общая точность (Accuracy): {accuracy_score(y_test, prognoz):.4f}")
print("\nДетальный отчет по классификации:\n", classification_report(y_test, prognoz))

# # График матрицы ошибок
# plt.figure(figsize=(7, 5))
# sns.heatmap(confusion_matrix(y_test, prognoz), annot=True, fmt='d', cmap='Blues')
# plt.title('Матрица ошибок (после доработки данных)')
# plt.show()

# График важности признаков
vazhnost = pd.Series(model_lesa.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=vazhnost, y=vazhnost.index, palette='magma', hue=vazhnost.index, legend=False)
plt.title('Важность факторов (включая Fatigue_Index и is_student)')
plt.tight_layout()
plt.show()
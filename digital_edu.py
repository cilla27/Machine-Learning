import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 

file_path = 'train.csv'
df = pd.read_csv(file_path) 

def check_null_values(df):
    null_counts = df.isnull().sum()
    print('\nJumlah nilai null di setiap kolom:')
    print(null_counts[null_counts > 0])

check_null_values(df)

if 'd' in df.columns:
    df.drop(columns=['d'], inplace=True)

def calculate_age(bdate):
    try:
        birth_year = int(str(bdate).split('.')[-1])
        return 2024 - birth_year if birth_year > 1900 else np.nan
    except:
        return np.nan

df['age'] = df['bdate'].apply(calculate_age)
df.drop(columns=['bdate'], inplace=True)

x = df.drop(columns=['result'])
y = df['result']

num_cols = x.select_dtypes(include=['int64', 'float64']).columns
cat_cols = x.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

x[num_cols] = num_imputer.fit_transform(x[num_cols])
x[cat_cols] = cat_imputer.fit_transform(x[cat_cols])

for col in cat_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])

scaler = StandardScaler()
x[num_cols] = scaler.fit_transform(x[num_cols])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
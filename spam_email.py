import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('email.csv')

encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])
df = df.drop_duplicates(keep='first')
df['num_char'] = df['Message'].apply(len)
x = df['Message']
y = df['Category']
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(x_train, y_train)

pickle.dump(model, open('spam123.pkl', 'wb'))
pickle.dump(cv, open('vec123.pkl', 'wb'))









import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st


data = pd.read_csv('spam.csv')
print(data.head())


data.drop_duplicates(inplace=True)

print(data.isnull().sum())



data['Category'] = data['Category'].replace(['ham' , 'spam'] , ['Not Spam' , 'Spam'])

print(data.head(5))


mess = data['Message']
cat = data['Category']

(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2, random_state=42)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)


#  Creating Model ----------------------------------------------------

model = MultinomialNB()
model.fit(features, cat_train)

# Test out Model ----------------------------------------------------

features_test = cv.transform(mess_test)
print("Accuracy: {}".format(model.score(features_test, cat_test)))


# Predicting ----------------------------------------------------

def predict(text):
    text_features = cv.transform([text])
    prediction = model.predict(text_features)[0]
    return prediction


st.header('Spam Detection')


res = predict("Hello, how are you?")

input_mess = st.text_input('Enter your message Here')

if st.button('Predict'):
    res = predict(input_mess)
    if res == 'Not Spam':
        st.success('Not Spam')
    else:
        st.error('Spam')
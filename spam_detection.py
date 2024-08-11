import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and preprocess data
data = pd.read_csv('spam.csv')

# Drop duplicates
data.drop_duplicates(inplace=True)

# Check for missing values
if data.isnull().sum().any():
    st.error("Data contains missing values. Please clean the data.")
else:
    st.write("Data loaded successfully.")

# Update category labels
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Display a sample of the data
st.write("### Sample Data:")
st.dataframe(data.head(5))

# Split data into training and testing sets
mess = data['Message']
cat = data['Category']
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

# Vectorize text data
cv = CountVectorizer(stop_words='english')
features_train = cv.fit_transform(mess_train)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(features_train, cat_train)

# Create a function to predict if a message is spam
def predict(text):
    text_features = cv.transform([text])
    prediction = model.predict(text_features)[0]
    return prediction

# Streamlit app UI
st.set_page_config(page_title="Spam Detection", layout="wide")
st.title('📧 Spam Detection')

# Sidebar for additional controls or information
st.sidebar.header('About')
st.sidebar.info(
    """
    This app uses a Naive Bayes classifier to detect if a given message is spam or not.
    - **Model**: Multinomial Naive Bayes
    - **Data**: SMS Spam Collection Dataset
    """
)

# Input form
st.subheader('Enter Your Message:')
input_mess = st.text_area('Message:', placeholder='Type your message here...')

if st.button('Predict'):
    if input_mess:
        result = predict(input_mess)
        if result == 'Not Spam':
            st.success('✅ **Not Spam**')
        else:
            st.error('🚫 **Spam**')
    else:
        st.warning('⚠️ Please enter a message to classify.')

# Add an image for a more engaging experience (optional)
st.image('https://www.example.com/your_image.png', caption='Spam Detection Model', use_column_width=True)

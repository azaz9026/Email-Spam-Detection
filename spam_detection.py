import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from PIL import Image , ImageOps, ImageDraw

# Configure the page
st.set_page_config(
    page_title="Spam Detection",
    page_icon=":envelope:",  # Or use a custom image path
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess data
data = pd.read_csv('spam.csv')

# Drop duplicates
data.drop_duplicates(inplace=True)

# Update category labels
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Display a sample of the data
st.write("### Sample Data:")
rows = st.slider("Number of rows to view:", min_value=1, max_value=20, value=5)
st.dataframe(data.head(rows))

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

# Title and Description
st.title("Spam Detection App")
st.write("This app uses a Naive Bayes classifier to detect if a message is Spam or Not Spam.")

# Sidebar for Input
st.sidebar.header("Input Message")
input_mess = st.sidebar.text_area('Enter your message here')

# Load the image
image = Image.open('logos.webp')  # Replace with your image path

# Create a circular mask
mask = Image.new("L", image.size, 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0) + image.size, fill=255)

# Apply the mask to the image
rounded_image = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
rounded_image.putalpha(mask)

# Display the rounded image
st.image(rounded_image, width=150)

# Predict Button
if st.sidebar.button('Predict'):
    res = predict(input_mess)
    confidence = model.predict_proba(cv.transform([input_mess]))[0]
    if res == 'Not Spam':
        st.sidebar.success(f'Prediction: Not Spam ({confidence[1]:.2f} confidence)')
    else:
        st.sidebar.error(f'Prediction: Spam ({confidence[0]:.2f} confidence)')

# Model Accuracy
features_test = cv.transform(mess_test)
accuracy = model.score(features_test, cat_test)
st.write(f"### Model Accuracy: {accuracy:.2%}")

# Example Prediction
st.write("### Example Prediction:")
example_text = st.text_input("Enter an example message:", "Hello, how are you?")
res = predict(example_text)
st.write(f"Message: {example_text}")
st.write(f"Prediction: {res}")

# Footer
st.markdown("---")
st.markdown("Created by [Your Name](https://your-link.com)")

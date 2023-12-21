import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Funky Movie Rating Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
def train_model_and_evaluate(data):
    features = ['Year', 'Duration', 'Votes']
    target = 'Rating'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    return model, accuracy
def predict_rating(model, user_input):
    predicted_rating = model.predict([user_input])
    predicted_rating = max(0, min(predicted_rating[0], 10))
    return round(predicted_rating, 1) 
def main():
    st.title('🍿 Funky Movie Rating Prediction 🎥')
    file_path = 'cleaned_data.csv'
    cleaned_data = load_data(file_path)
    model, accuracy = train_model_and_evaluate(cleaned_data)
    st.sidebar.title('🎬 Enter movie details:')
    user_input = {}
    for column in ['Name', 'Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
        user_input[column] = st.sidebar.text_input(column, f"Enter {column}", key=column)
        if user_input[column] == f"Enter {column}":
            user_input[column] = ""
    if st.sidebar.button('Predict'):
        error = False
        error_messages = []
        numeric_columns = ['Year', 'Duration', 'Votes']
        for col in numeric_columns:
            if user_input[col] == "":
                error = True
                error_messages.append(f"{column} field is empty")
            else:
                try:
                    user_input[col] = float(user_input[col])
                except ValueError:
                    error = True
                    error_messages.append(f"{column} field should be a number")
        if error:
            st.warning("Please resolve the following issues:")
            for msg in error_messages:
                st.write(f"- {msg}")
        else:
            st.write('🍿 Entered movie details:')
            for key, value in user_input.items():
                if isinstance(value, float):
                    value = round(value, 1)
                st.write(f"{key}: {value}")
            predicted_rating = predict_rating(model, [user_input[col] for col in ['Year', 'Duration', 'Votes']])
            st.write(f"🌟 The predicted rating for the movie is: {predicted_rating}")
            st.write(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
if __name__ == '__main__':
    main()

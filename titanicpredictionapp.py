import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv('titanicdataset.csv')
data['Sex'] = data['Sex'].astype(str)
data['Pclass'] = data['Pclass'].astype(str)
st.title('Titanic Survival Prediction')
st.sidebar.subheader('Training Data Summary')
st.sidebar.write(data.describe())
def prepare_user_input(user_df, input_features):
    user_encoded = pd.DataFrame(0, index=user_df.index, columns=input_features)
    for col in user_df.columns:
        col_name = f"{col}_{user_df[col].iloc[0]}"
        if col_name in input_features:
            user_encoded[col_name] = 1
    
    return user_encoded

def user_input():
    st.sidebar.subheader('User Input')

    pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
    sex = st.sidebar.radio('Sex', ['male', 'female'])
    age = st.sidebar.slider('Age', 0, 80, 30)
    sibsp = st.sidebar.slider('SibSp', 0, 8, 2)
    parch = st.sidebar.slider('Parch', 0, 6, 1)

    user_data = {
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch
    }
    return pd.DataFrame(user_data, index=[0])

user_df = user_input()
X = data.drop(['Survived'], axis=1)
y = data['Survived']
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
input_features = encoder.get_feature_names_out(X.columns)
X_df = pd.DataFrame(X_encoded.toarray(), columns=input_features)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_df)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
st.sidebar.subheader('Model Accuracy')
st.sidebar.write(f"Model Accuracy: {accuracy_score(y_test, rf.predict(X_test)) * 100:.2f}%")
st.subheader('Age Distribution')
fig_age, ax_age = plt.subplots(figsize=(8, 6))
sns.histplot(data['Age'], bins=20, ax=ax_age)
ax_age.set_xlabel('Age')
ax_age.set_ylabel('Frequency')
st.pyplot(fig_age)
class_counts = data['Pclass'].value_counts().sort_index()
st.subheader('Passenger Class Distribution')
fig_pclass, ax_pclass = plt.subplots()
ax_pclass.bar(class_counts.index.astype(str), class_counts.values)
ax_pclass.set_xlabel('Pclass')
ax_pclass.set_ylabel('Count')
st.pyplot(fig_pclass)
survival_by_pclass = data.groupby('Pclass')['Survived'].mean().sort_index()
st.subheader('Survival Rate by Passenger Class')
fig_survival_pclass, ax_survival_pclass = plt.subplots()
survival_by_pclass.plot(kind='bar', stacked=True, ax=ax_survival_pclass)
ax_survival_pclass.set_xlabel('Pclass')
ax_survival_pclass.set_ylabel('Survival Rate')
st.pyplot(fig_survival_pclass)
gender_counts = data['Sex'].value_counts()
st.subheader('Gender Distribution')
fig_gender, ax_gender = plt.subplots()
ax_gender.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
ax_gender.axis('equal')
st.pyplot(fig_gender)
survival_by_gender = data.groupby(['Sex', 'Survived']).size().unstack()
st.subheader('Survival by Gender')
fig_survival_gender, ax_survival_gender = plt.subplots()
survival_by_gender.plot(kind='bar', stacked=True, ax=ax_survival_gender)
ax_survival_gender.set_xlabel('Gender')
ax_survival_gender.set_ylabel('Passenger Count')
st.pyplot(fig_survival_gender)
user_encoded = prepare_user_input(user_df, input_features)
user_imputed = imputer.transform(user_encoded)
prediction = rf.predict(user_imputed)
st.sidebar.subheader('PREDICTION')
if prediction[0] == 1:
    st.sidebar.write("The passenger is predicted to be survived.")
else:
    st.sidebar.write("The passenger is predicted not to be survive.")

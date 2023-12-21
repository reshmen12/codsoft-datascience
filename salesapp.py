import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('advertising.csv')

feature_names = ['TV', 'Radio', 'Newspaper']
X = data[feature_names]
y = data['Sales']
model = LinearRegression()
model.fit(X, y)

st.title('Sales Prediction Tool')
tv_spend = st.sidebar.slider('Enter TV Spend', min_value=0.0,max_value=500.0)
radio_spend = st.sidebar.slider('Enter Radio Spend', min_value=0.0,max_value=500.0)
newspaper_spend = st.sidebar.slider('Enter Newspaper Spend', min_value=0.0,max_value=500.0)

if st.button("view Data analysis"):
        st.subheader('Pie Chart: Distribution of Sales by Advertising Medium')
        pie_data = data[['TV', 'Radio', 'Newspaper']].sum()
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99'])
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

        st.subheader('Box Plot: Spread of Sales for Each Advertising Medium')
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data[['TV', 'Radio', 'Newspaper']], ax=ax_box, palette=['#FF9999', '#66B2FF', '#99FF99'])
        ax_box.set_ylabel('Sales')
        ax_box.set_xlabel('Advertising Medium')
        ax_box.set_title('Spread of Sales for Each Advertising Medium')
        st.pyplot(fig_box)
        st.subheader('Histograms: Distribution of Spends')
        fig_hist, ax_hist = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        for i, column in enumerate(['TV', 'Radio', 'Newspaper']):
                sns.histplot(data[column], ax=ax_hist[i], color='#27DBFB', bins=20)
                ax_hist[i].set_title(f'{column} Spend Distribution')
                ax_hist[i].set_xlabel(f'{column} Spend')
                ax_hist[i].set_ylabel('Frequency')
        st.pyplot(fig_hist)
        st.subheader('Heatmap: Correlation Matrix')
        corr_matrix = data.corr()
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig_heatmap)
input_data = [[tv_spend, radio_spend, newspaper_spend]]
predicted_sales = model.predict(input_data)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
if st.button('Predict'):
    st.write(f"Predicted Sales for the future trend:${predicted_sales[0]:.2f}")
    st.write(f"R-squared on the entire dataset: {r2*100:.2f}")
    y_pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, color='blue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.set_xlabel('Actual Sales')
    ax.set_ylabel('Predicted Sales')
    ax.set_title('Actual vs Predicted Sales')
    st.pyplot(fig)


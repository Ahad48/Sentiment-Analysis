import streamlit as st
from news import articles, print_bar_graph, plot_word_cloud

st.title("Entity News Sentiment Analysis")
st.write(
    "A machine learning app to predict the sentiment of the news articles that are published"
)

form = st.form(key="my_form")
entity = form.text_input(label="Enter the text of your movie entity")
submit = form.form_submit_button(label="Find News")

if submit:
    st.write(f"Searching news articles for: {entity}")
    df = articles(entity)
    st.dataframe(df)
    st.plotly_chart(print_bar_graph(df))
    st.pyplot(plot_word_cloud(df))

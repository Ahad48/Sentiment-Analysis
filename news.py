import requests
import numpy as np
import pandas as pd
from newspaper import Article
from newspaper import Config
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random
from wordcloud import WordCloud

# Import the functions from the sentiment_functions python file
from sentiment_functions import preprocess_text, load_models

# Import parallel processing module
from multiprocesspandas import applyparallel


user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent


## News API KEY 
apiKey = '9e33b380b35543bf85a8d249bd209fd4'



def make_predictions(df):
    max_length = 250
    text = tokenizer.texts_to_sequences(df)
    text = pad_sequences(text, maxlen=max_length, padding='post', truncating= 'post')
    results = model.predict(text).reshape(-1)
    return lb.inverse_transform(results),results
    


def get_article(url):
    #Get the url
    article = Article(url,config=config)

    #Download the article
    article.download()

    #parse the article
    article.parse()

    #return the text
    return article.text

def search_entity(entity,language = 'en',apiKey = apiKey):
    #Send a REST request, more parameters can be added
    search = f"https://newsapi.org/v2/top-headlines?q={entity}&language={language}&apiKey={apiKey}"
    articles_api = requests.get(search).json()

    #Converting the articles into a dataframe
    articles_df = pd.DataFrame.from_dict(articles_api.get('articles'))
    try:
        articles_df['Text'] = articles_df.url.apply(get_article)
        # Apply Preprocessing to the data
        articles_df['Clean_text'] = articles_df["Text"].apply_parallel(preprocess_text,num_processes = 4)
    except:
        articles_df['Clean_text'] = articles_df["content"].apply_parallel(preprocess_text,num_processes = 4)
    return articles_df

def articles(entity):
    data = search_entity(entity)
    data['source'] = data['source'].apply(lambda x:x.get('name'))
    predictions = make_predictions(data['Clean_text'])
    return pd.DataFrame(list(zip(data['Text'],predictions[0],predictions[1],data['url'])),columns=['Text','Sentiment','Sentiment Score','URL'])


model, lb, tokenizer = load_models('models/basic_keras_model.h5')


def print_bar_graph(articles):
    articles_dict = articles.groupby("Sentiment").size().to_dict()

    fig = go.Figure()

    fig.add_trace(go.Bar(x=list(articles_dict.keys()), y=list(articles_dict.values())))
    fig.update_traces(marker_color='rgb(153, 204, 255)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)

    fig.update_layout(title='Sentiment Graph of Articles')
#     fig.show()
    return fig




def plot_word_cloud(articles):
    def color(word, font_size,font_path, position, orientation, random_state=None):
        return "hsl(199, 100%%, %d%%)" % random.randint(10, 40)

    all_words_pros = ' '.join(articles['Text'])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color='white').generate(all_words_pros)
    plt.figure(figsize=(20, 14))
    plt.imshow(wordcloud.recolor(color_func=color, random_state=3), interpolation="bilinear")
    plt.axis('off')
    plt.title('Sentiment Cloud',fontsize=50)
#     plt.show()
    return plt






import pandas as pd
from scipy import sparse
import numpy as np
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import altair as alt
from PIL import Image
from matplotlib import interactive
import warnings
import random
import cv2
from imageio import imread
from skimage.transform import resize
from sklearn.cluster import KMeans
from matplotlib.colors import to_hex

interactive(True)
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
     page_title="Ka|Ve Youtube APP",
     page_icon=":shark:",
     layout="centered",
     initial_sidebar_state="auto",
 )


image = Image.open('kave.png')
resim = Image.open("resim.png")
st.image(image, caption='Karmaşık Sistemler ve Veri Bilimi Topluluğu')
st.header("Youtube Analiz ve Tavsiye Sistemi")
st.subheader('Biz Ne Yaptık?\nNeyi Amaçlıyoruz')
st.text("""
    Öncelikle bizler yazılım sektörünün ürün aşamasında fazlaca önemli olduğunu düşündüğümüz
    Analiz ve Tavsiye üzerine bir araştırma yaptık. Bu araştırma sonucunda şunu fark ettik:
    Büyük şirketler stratejik kararlar alırken ya da ürünü müşteriye ulaştırırken bu sistemleri
    fazlaca kullanıyor. Netflix, Amazon vs.
    
    --
    Bu projede de biz youtube macerasına atılacak kullanıcıları hedef alan bir ürün geliştirdik.

    Bu üründe örnek video tavsiyeleri renk tavsiyeleri ve sektörel bazda
    sonuçlara ulaşabileceği grafikleri kullanıcılarımızın önüne serdik.

    Burada amacımız müşterilerimizin başarıya giden yolda attığı adımlara destek
    olmaktır.


    """)



st.image(resim)

with open('uzantilar.txt') as f:
    lines = f.readlines()

animal     = list(filter(lambda k: 'animal' in k, lines))
art        = list(filter(lambda k: 'art' in k, lines))
automotive = list(filter(lambda k: 'automotive' in k, lines))
books      = list(filter(lambda k: 'books' in k, lines))
coding     = list(filter(lambda k: 'coding' in k, lines))
comedy     = list(filter(lambda k: 'comedy' in k, lines))
cooking    = list(filter(lambda k: 'cooking' in k, lines))
diy        = list(filter(lambda k: 'diy' in k, lines))
education  = list(filter(lambda k: 'education' in k, lines))
language   = list(filter(lambda k: 'language' in k, lines))
makeup     = list(filter(lambda k: 'makeup' in k, lines))
music      = list(filter(lambda k: 'music' in k, lines))
sport      = list(filter(lambda k: 'sport' in k, lines))
technology = list(filter(lambda k: 'technology' in k, lines))
youtubers  = list(filter(lambda k: 'youtubers' in k, lines))
option = st.selectbox('Which category',
    ["Animal","Art","Automotive","Books","Coding","Comedy","Cooking",
    "Diy","Education","Language","Makeup","Music","Sport","Technology"
    "Youtubers"])
if(option=="Animal"):option=animal
elif(option=="Art"):option=art
elif(option=="Automotive"):option=automotive
elif(option=="Books"):option=books
elif(option=="Coding"):option=coding
elif(option=="Comedy"):option=comedy
elif(option=="Cooking"):option=cooking
elif(option=="Diy"):option=diy
elif(option=="Education"):option=education
elif(option=="Language"):option=language
elif(option=="Makeup"):option=makeup
elif(option=="Music"):option=music
elif(option=="Sport"):option=sport
elif(option=="Technology"):option=technology
elif(option=="Youtubers"):option=youtubers

for i in range(5):
    try:
        n = random.randint(0,len(option))
        temp = option[n]
        temp = temp.rstrip('\n')
        break
    except:
        pass

print(temp)

def plot_colors( centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    for (percent, color) in enumerate(centroids):
        color *=255
        endX = startX + 50
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    return bar

np.random.seed(0)
filepath = 'images/'+temp
img1 = Image.open(filepath)
img = imread(filepath)
st.image(img1, caption = "Size Önerilen Resim")
st.info("Eğer Resmi Beğenmediyseniz Tekrardan Kategori Seçme Yerinden Seçtiğiniz Kategoriyi Seçiniz")
img = resize(img, (200, 200))
data = pd.DataFrame(img.reshape(-1, 3),
                    columns=['R', 'G', 'B'])
kmeans = KMeans(n_clusters=6,
                random_state=0)
data['Cluster'] = kmeans.fit_predict(data)
palette = kmeans.cluster_centers_
palette_list = list()
for color in palette:
    palette_list.append([[tuple(color)]])
hexkod =[]
for color in palette_list:
    #st.write(to_hex(color[0][0]))
    hexkod.append(to_hex(color[0][0]))
st.write(hexkod)
data['R_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][0])
data['G_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][1])
data['B_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][2])
img_c = data[['R_cluster', 'G_cluster', 'B_cluster']].values
img_c = img_c.reshape(200, 200, 3)
img_c = resize(img_c, (800, 1200))
bar = plot_colors( kmeans.cluster_centers_)
st.image(bar, caption='Kullanabileceğiniz Renk Barı')


st.image(resim)



def token_def(values, stop_word_list):
        filtered_words = [word for word in values.split() if word not in stop_word_list]
        not_stopword_doc = " ".join(filtered_words)
        return not_stopword_doc

def recommend_system(df):
    ### FEATURE ENGINEERING KISMI
    df['views_list'] = df['views_list'].str[:-11]
    df['years_list'] = df['years_list'].str[:-7]
    df.rename(columns = {"years_list" : "months_before"}, inplace = True)
    df.months_before.replace(["1 y", "2 y", "3 y", "4 y", "5 y", "6 y", "7 y", "8 y",
                             "9 y", "10 y", "11 y", "12 y", "13 y", "14 y", "10 ay önce yay", "2 haf"],[12,24,36,48,60,72,84,
                                                                                                       96,108,120,132,144,156,168,
                                                                                                       10, 1], inplace = True)
    liste = list()
    for i in df.views_list.values:
        i = i.replace(",",".")
        if "\xa0Mn " in i and "." in i:
            i = i.replace("\xa0Mn ", "00000")
        elif "\xa0Mn " in i and "." not in i:
            i = i.replace("\xa0Mn ", "000000")           ### izlenmeleri sayısal değerlerle değiştiriyorum.
        elif "\xa0B " in i:
            i = i.replace("\xa0B ", "000")
        i = i.replace(".","")
        liste.append(i)
        
    df.views_list = liste
    df.views_list = df.views_list.astype(int)
    df.months_before = df.months_before.astype(int)
    df["views_per_months"] = df.views_list / df.months_before         
    df.drop(['Unnamed: 0', 'Unnamed: 0.1',
           'views_list', 'months_before', 'link_list', 'photo_link_list'], 1, inplace = True)

    liste = list()
    for i in df.views_per_months.values:
        if i <= 10000:
            liste.append(1)
        elif i <= 100000:
            liste.append(2)                        ### views_per_month'a göre her videoya bir popülerlik katsayısı atıyorum.
        elif i <= 1000000:
            liste.append(3)
        else:
            liste.append(4)
    df.views_per_months = liste
    df.fillna("", inplace = True)


    ### NLP Çalışmaları
    stop_word_list = nltk.corpus.stopwords.words('english')
    docs = df['descriptions']
    docs = docs.map(lambda x: re.sub(r"[-()\"#/@;:<>{}+=~|.!?,]", '', x))
    docs = docs.map(lambda x: x.lower())
    docs = docs.map(lambda x: x.strip())
    docs = docs.map(lambda x: token_def(x,stop_word_list))

    df['descriptions'] = docs
    df['channel_name'] = df['channel_name'].map(lambda x: x.split(','))
    df['category'] = df['category'].map(lambda x: x.split(','))
    df['descriptions'] = df['descriptions'].map(lambda x: x.split(','))

    for index, row in df.iterrows():
        row['category'] = [x.lower().replace(' ',' ') for x in row['category']]
        row['channel_name'] = [x.lower().replace(' ',' ') for x in row['channel_name']]
        row['descriptions'] = [x.lower().replace(' ',' ') for x in row['descriptions']]

    df["our_coef"] = df.views_per_months * df.category
    df.drop(["views_per_months", "category"], 1, inplace = True)
    df['Bag_of_words'] = ''
    columns = ['our_coef', 'channel_name', 'descriptions']

    for index, row in df.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        row['Bag_of_words'] = words
        
        
    df['Bag_of_words'] = df['Bag_of_words'].str.strip().str.replace('   ', '  ').str.replace('  ', ' ')
    df = df[['title_list','Bag_of_words']]

    ######## Kullanıcı için olan kısım (inputlar vs)

    st.write("Araştırma Yapmak İstediğiniz Başlık Nedir? \n")
    input_title = st.text_input('title')
    st.write("Daha İyi Sonuçlar Bulabilmek Adına Anahtar Birkaç Anahtar Kelime Söyler misiniz\n")
    input_keywords = st.text_input('keywords')
    df.loc[len(df)-1]["Bag_of_words"] = input_title
    df.loc[len(df)-1]["title_list"] = input_keywords

    count = CountVectorizer()
    count_matrix = count.fit_transform(df['Bag_of_words'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df['title_list'])
    return df, input_keywords, cosine_sim, indices 

def recommend(df, title, cosine_sim, indices):
    recommended_links = []
    idx = indices[indices == title].index[0]   # to get the index of the movie title matching the input movie
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)   # similarity scores in descending order
    top_6_indices = list(score_series.iloc[1:6].index)   # to get the indices of top 10 most similar movies
    # [1:11] to exclude 0 (index 0 is the input movie itself)
    
    for i in top_6_indices:   # to append the titles of top 10 similar movies to the recommended_movies list
        recommended_links.append(list(df['title_list'])[i])
    df = df[:-1]    # atıyoruz kullanıcının girdiği kısmı.
    return recommended_links











# recommendation kısmı 

def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x
    if 'B' in x:
        if len(x) > 1:
            return float(x.replace('B', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    return 0.0

def average_dataset(df):
    
    df.drop(columns = ["Unnamed: 0", "Unnamed: 0.1", "link_list", "photo_link_list"], inplace = True)
    df['views_list'] = df['views_list'].map(lambda x: x.rstrip(' görüntüleme'))
    df['views_list'] = df['views_list'].str.replace(',', '.')
    df['views_list'] = df['views_list'].apply(value_to_float)
    df['views_list'] = df['views_list'] / 1000
    df['views_list'] = df['views_list'].astype(int)
    avg_df = df.groupby('category').mean()
    avg_df.reset_index(inplace=True)
    return avg_df

def draw_barplot_avg_data(data,x,y):

    df = average_dataset(data)
    a=plt.figure(figsize=(10,10))
    a=plt.xticks(rotation='vertical')
    a=sns.barplot(x=x, y=y, data=df, order=df.sort_values(y)[x], palette=("Set3"))
    #plt.show()
    st.pyplot(plt.show())

# views lerin plot grafiği 

def nlp_title_dataset(df):


    X_df = df["title_list"]
    corpus = [word_tokenize(token) for token in X_df]
    lowercase_title = [[token.lower() for token in doc] for doc in corpus]
    alphas = [[token for token in doc if token.isalpha()] for doc in lowercase_title]
    stop_words = stopwords.words('english')
    title_no_stop = [[token for token in doc if token not in stop_words] for doc in alphas]
    stemmer = PorterStemmer()
    stemmed = [[stemmer.stem(token) for token in doc] for doc in title_no_stop]
    title_clean_str = [ ' '.join(doc) for doc in stemmed]
    #Number of words
    nb_words = [len(tokens) for tokens in alphas]
    #Number of unique words
    alphas_unique = [set(doc) for doc in alphas]
    nb_words_unique = [len(doc) for doc in alphas_unique]
    #Number of characters
    train_str = [ ' '.join(doc) for doc in lowercase_title]
    nb_characters = [len(doc) for doc in train_str]
    #Number of stopwords
    train_stopwords = [[token for token in doc if token in stop_words] for doc in alphas]
    nb_stopwords = [len(doc) for doc in train_stopwords]
    #Number of punctuations
    non_alphas = [[token for token in doc if token.isalpha() == False] for doc in lowercase_title]
    nb_punctuation = [len(doc) for doc in non_alphas]
    df_clean = pd.DataFrame(data={'title_clean': title_clean_str})
    df_category = pd.DataFrame(data={'category': df['category']})
    nb_words = pd.Series(nb_words)
    nb_words_unique = pd.Series(nb_words_unique)
    nb_characters = pd.Series(nb_characters)
    nb_stopwords = pd.Series(nb_stopwords)
    nb_punctuation = pd.Series(nb_punctuation)
    df_show = pd.concat([df_category, df_clean, nb_words, nb_words_unique, nb_characters, nb_stopwords, nb_punctuation], axis=1).rename(columns={
    0: "nb_words", 1: 'nb_words_unique', 2: 'nb_characters', 3: 'nb_stopwords', 4: 'nb_punctuation'
    })
    mean_df = df_show.groupby('category').mean()
    mean_df.reset_index(inplace=True)
    return mean_df

def draw_barplot_nlp_title_data(df,x1,y1):
    dataframe = nlp_title_dataset(df)
    print(dataframe.columns)
    plt.figure(figsize=(10,10))
    plt.xticks(rotation='vertical')
    sns.barplot(x ="category", y = "nb_punctuation",data=dataframe, order=dataframe.sort_values(y1)[x1])
    st.pyplot(plt.show())


#dağılımlı izlenme grafiği
def draw_graph(data ,x,y): 
    data =data_prepocessing(data)
    fig = go.Figure(data=go.Scatter(x=data['category'],
                                    y=data['views_list'],
                                    mode='markers',
                                    marker_color=data['views_list'],
                                    text=data['title_list'])) # hover text goes here

    fig.update_layout(title='Kategorilerin izlenme ve adları')
    st.plotly_chart(fig)

def data_prepocessing(data):

    #data.drop(columns = ["link_list", "photo_link_list"], inplace = True)
    data['views_list'] = data['views_list'].map(lambda x: x.rstrip(' görüntüleme'))
    data['views_list'] = data['views_list'].str.replace(',', '.')
    data['views_list'] = data['views_list'].apply(value_to_float)
    data['years_list'] = data['years_list'].map(lambda x: x.rstrip(' yıl önce,a,haft,ay önce yayınland'))
    data['years_list'] = data['years_list'].str.replace(',', '.')

    data["views_list"] = data["views_list"].astype(int)
    return data

data = pd.read_excel ('output2.xlsx')
df = pd.DataFrame(data)
df1 =df.copy()
df2 =df.copy()
#recommendation system -> çalışıyor inputu data frame olarak çıkartabiliriz
df1 , input_keywords, cosine_sim , indices1= recommend_system(df1)
st.write(recommend(df1, input_keywords, cosine_sim, indices1))
st.image(resim)

#çalışıyor
draw_barplot_nlp_title_data(df=df, x1="category", y1="nb_punctuation")
st.image(resim)

#çalışıyor
draw_barplot_avg_data(x="category", y="views_list", data=df)
st.image(resim)

#çalışıyor
draw_graph(df2, x='category',y='views_list')
st.image(resim)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests
import datetime
import pandas as pd
import json
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords
import re
#import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: # This is the correct exception to catch when a resource is not found
    nltk.download('stopwords')
    print("NLTK 'stopwords' downloaded successfully.")
except Exception as e:
    print(f"An unexpected error occurred during NLTK stopwords check/download: {e}")

fields = "name,cca2" 
response = requests.get(f"https://restcountries.com/v3.1/all?fields={fields}")
st.set_page_config(page_title="streamlit News App", page_icon=":newspaper:", layout="wide")
#st.title("                📰:rainbow[ News and Sentiment] 🗞️                ")
#st.markdown("<h1 style='text-align: center; color: #1f77b4;'>📰News and Sentiment🗞️</h1>", unsafe_allow_html=True)
search_text=""
country_code=""
stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
   
    if pd.isna(text) or not isinstance(text, str): # Handle potential NaN or non-string values
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0, 'overall_sentiment': 'Neutral (N/A)'}
    #if isinstance(text,str):
       # words=text.lower().split()
        #words=[word for word in words if word not in stop_words and word not in string.punctuation]
       # processed_text = " ".join(words)
    processed_texts=preprocess_text(text)
    scores = sid.polarity_scores(processed_texts)
    compound_score = scores['compound']
  
    # Classify overall sentiment based on compound score thresholds
    if compound_score >= 0.05:
        overall_sentiment = "Positive"
    elif compound_score <= -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return {
        'Negative': scores['neg'],
        'Neutral': scores['neu'],
        'Positive': scores['pos'],
        'Compound': compound_score,
        'Overall_sentiment': overall_sentiment
    }
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    words = text.split()
   # stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)
def intro_page():
     #st.markdown("<h1 style='text-align: center;'>📰 Welcome to the News and Sentiment Analyzer! 🗞️</h1>", unsafe_allow_html=True) 
    st.title("📰 _:rainbow[Welcome to the News and Sentiment Analyzer!] 🗞️_")

    st.write("---")

    st.header(":red[Dive Deep into the World of News]🌐")
    st.markdown(
        """
        This interactive **Streamlit News and Sentiment Analyzer** is your gateway to exploring global news with insightful analytics.
        Designed for **curious minds, researchers, and anyone keen on understanding public sentiment**, this application goes beyond just fetching headlines. It empowers you to:
        * **:rainbow[Search for News:]** Effortlessly find articles from various **countries** and **categories**, or drill down with specific **keywords**.
        * **:rainbow[Uncover Sentiment:]** Get an immediate understanding of the **emotional tone** (positive, negative, or neutral) behind news articles, helping you grasp the prevailing sentiment on any given topic.
        * **:rainbow[Discover Hidden Patterns:]** Utilize advanced **clustering** to group similar news articles, revealing underlying themes and connections you might otherwise miss.
        * **:rainbow[Generate Comprehensive Reports:]** Access **yearly news reports** that visualize sentiment trends, top categories, and frequent keywords, providing a macro-level view of the news landscape.
        """
    )
    st.write("---")

    st.header(" :rainbow[How It Works: A Glimpse Behind the Headlines] ⚙️")
    st.markdown(
        """
        At its core, this application leverages powerful technologies to bring you a seamless experience:
        * **:rainbow[GNews API Integration:]** Fetches real-time and historical news articles from a vast array of sources.
        * **:rainbow[Natural Language Processing (NLP):]** Employs **NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)** for nuanced sentiment analysis, processing text to determine its emotional charge.
        * **:rainbow[Machine Learning for Clustering:]** Utilizes **TF-IDF Vectorization** and **K-Means Clustering** to group news articles based on their textual content, identifying key topics and narratives.
        * **:rainbow[Interactive Visualizations:]** Presents complex data in an easy-to-understand format using **Matplotlib** and **WordCloud**, making trends and patterns immediately apparent.
        """
    )

    st.write("---")
    st.header(" :rainbow[Key Features at Your Fingertips] 👇")
    st.markdown(
        """
        * **:rainbow[Flexible News Search:]** Filter news by **country**, **category (Business, Entertainment, Health, Science, Sports, Technology, Headlines)**, or by **custom keywords**.
        * **:rainbow[Detailed Sentiment Breakdown:]** For each article, view scores for **negative, neutral, positive, and compound sentiment**, along with an **overall sentiment classification**.
        * **:rainbow[Dynamic News Clustering:]** Input a custom query and instantly see which news cluster it belongs to, along with its sentiment analysis.
        * **:rainbow[Annual Reporting:]** Generate insightful reports for selected years, featuring:
            * **:blue[Overall Sentiment Distribution Pie Chart]** 📊
            * **:blue[Monthly Sentiment Trend Line Chart]** 📈
            * **:blue[Top Categories by Volume Bar Chart]** 🏆
            * **:blue[Most Frequent Keywords Word Cloud]** 📝
            * **:blue[Average Sentiment by Category Bar Chart]** 🌈
        """
    )
    st.markdown(
        """
        This app is designed to transform how you consume and analyze news, providing a deeper, data-driven perspective on current events.
        Explore, analyze, and understand the news like never before!
        """
    )

   
def get_news(country,category, from_date, to_date, api_key,country_name):
    
    API_KEY = "358b617c0fc1638333b6c9d449fae32f" 
    

    if category == 'headlines':
       url= "https://gnews.io/api/v4/top-headlines"
       params = {
           "token": API_KEY,
           "q":f"{category} AND {country_name}",
           "category": category,
           "country": country,
           
           "lang": "en",  
           "max": 3   
       } 
    elif category =="search by keyword":  
       
       url = "https://gnews.io/api/v4/search"
       #from_date_iso = from_date.isoformat() + "T00:00:00Z"
      # to_date_iso = to_date.isoformat() + "T23:59:59Z" 
       
       if country == '':
           country=""
           
       params = {
           "token": API_KEY,
           "q":preprocess_text(search_text),
           #"category": category,
           "country": country,
          # "from": from_date_iso,
         #  "to": to_date_iso,
           "lang": "en",  
           "max": 3     
       }  
       
    else:    
       url = "https://gnews.io/api/v4/search"
       from_date_iso = from_date.isoformat() + "T00:00:00Z"
       to_date_iso = to_date.isoformat() + "T23:59:59Z"  
   
       params = {
           "token": API_KEY,
           "q":f"{category} AND {country_name}",
           "category": category,
           "country": country,
           "from": from_date_iso,
           "to": to_date_iso,
           "lang": "en",  
           "max": 3  
       }
    try:    
        response = requests.get(url, params=params)
        
        data = response.json()
      
    
    except Exception as e:
        st.error(f"An error occurred while generating the report: {e}")
        
    all_data=[]    
  
    if 'articles' in data:
      
       
        items=data['articles']
       
        for item in items:
           
           news_info={'link':item['url'],
                      'Headline':item['title'],
                      'Country':country_name,
                      'Category':category,
                      'Short_description':item['description'],
                      'Date':item['publishedAt']}
          
           all_data.append(news_info) 
          
           
        return all_data  
    else:
        st.write("Error fetching news:", data)
        return []
  
intro = st.sidebar.radio('Main Menu',["**_:blue[Introduction]_**","**_:blue[Check In for News App]_**"])
if  intro== "**_:blue[Introduction]_**":    
    #main_frame()  
    intro_page()
elif intro == "**_:blue[Check In for News App]_**":  
    #st.title("📖 _:orange[News App]_")

   # st.header("🗞️ :blue[News and Sentiment]", divider="rainbow")


    selected_question = st.sidebar.selectbox( "**_:rainbow[Select a Question to View Analysis]_**", ['Search News','Cluster','Report'],index=None)
    if selected_question == 'Search News':
               # Check if the request was successful
              
              if response.status_code == 200:
                      # Successful response, let's proceed
                      data = response.json()
                      

                     # Map country names to their corresponding 2-letter Alpha-2 codes
                      country_codes = {country['name']['common']: country['cca2'] for country in data}
                      sorted_countries = sorted(country_codes.keys())
                      selected_country = st.sidebar.selectbox(':rainbow[Select a country:]', sorted_countries,index=None)
                     # Get the corresponding 2-letter code for the selected country
                      if selected_country is not None:
                             country_code = country_codes[selected_country]
                            
                      categories = ['All','Business', 'Entertainment','HeadLines', 'Health', 'Science', 'Sports', 'Technology','Search by keyword']
                      selected_category = st.sidebar.selectbox(':rainbow[Select a category]', categories,index=None)
                       

  
                      max_date = datetime.date.today() 
                      min_date = datetime.date(1800, 1, 1)
                      from_date = st.sidebar.date_input(":rainbow[From date:]", min_value=min_date, max_value=max_date,key="start_date")
                      to_date = st.sidebar.date_input(":rainbow[To date:]", min_value=min_date, max_value=max_date, key="end_date")
                      if selected_category == "Search by keyword":
                          
                          search_text=st.sidebar.text_input(":rainbow[Enter text for search]") 
                      
                      search=st.sidebar.button("**_:rainbow[SEARCH]_**")
                      if (search):
                          
                          if country_code is not None and selected_category is not None:
                               #dff=pd.DataFrame(get_news(country_code.lower(),selected_category.lower(),from_date,to_date,"ac906e1373b34ad2ba54c1bae3c360ca",selected_country))
                               dff=pd.DataFrame(get_news(country_code.lower(),selected_category.lower(),from_date,to_date,
                                                                 "ac906e1373b34ad2ba54c1bae3c360ca",selected_country))
                                                                 
                               if  not dff.empty:
                                    sentiment_results = dff['Short_description'].apply(lambda x: pd.Series(get_vader_sentiment(x)))
                                
                                    dff = pd.concat([dff, sentiment_results], axis=1)


                                    dff['Links'] = dff['link']
                                    dff['Links'] = dff.apply(lambda row: f"<a href='{row['link']}' target='_blank'>{row['Headline']}</a>", axis=1)
                                   
                                    df_display = dff[['Links','Headline', 'Country', 'Category','Short_description','Date','Negative','Positive','Neutral','Compound','Overall_sentiment']]

                                    html_table = df_display.to_html(escape=False)
                                    st.markdown(html_table, unsafe_allow_html=True)

                                    st.write("---")

                               else:
                                    st.write(dff)
                          else:
                              st.write("Please select country and category")
                             
                           

              else:
                      st.error(f"Error fetching data: {response.status_code}")
                      st.write("Response content:", response.text)  # Print the error message if any
    elif selected_question == 'Cluster':    
              dfvec=pd.read_csv(r'c:\users\91904\combined.csv')
              vect=TfidfVectorizer(stop_words='english',max_features=5000)
              dfvec['headline'] = dfvec['headline'].fillna('')
              dfvec['short_description'] = dfvec['short_description'].fillna('')
              dfvec['text_content'] = dfvec['headline'] + " " + dfvec['short_description']
              x=vect.fit_transform(dfvec['text_content'])
              num_cluster=7
              kmean=KMeans(n_clusters=7,random_state=42,init='k-means++')
              dfvec['cluster']=kmean.fit_predict(x)
              cluster_name_mapping = {
                                    0: "US Presidential Campaigns (Clinton-Sanders Era)",
                                    1: "Regional US News & Local Stories",
                                    2: "Presidential & Judicial Affairs (Obama Era Focus)",
                                    3: "US Congressional & White House Politics",
                                    4: "Broad & General News Coverage",
                                    5: "Healthcare Policy & Debates",
                                    6: "Donald Trump & Related Politics"}
              unique_clusters = sorted(dfvec['cluster'].unique())
              n_clusters = len(unique_clusters)
              n_cols = min(3, n_clusters) 
              n_rows = (n_clusters + n_cols - 1) // n_cols
              #fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4)) 
              fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
              if n_rows * n_cols == 1: # Only one subplot
                                axes = np.array([axes]) # Wrap the single Axes object in an array
              else:
                                axes = axes.flatten() 
             # plt.figure(figsize=(n_cols * 5, n_rows * 4))
             
              for i, cluster_id in enumerate(unique_clusters):
    # Filter text content for the current cluster
                     cluster_text = " ".join(dfvec[dfvec['cluster'] == cluster_id]['text_content'].dropna())
                  
    # Generate word cloud
                     wordcloud = WordCloud(width=800, height=400, background_color='white',
                     max_words=50, collocations=False).generate(cluster_text)
                    
    # Get the meaningful title for the current cluster
                     cluster_title = cluster_name_mapping.get(cluster_id, f"Cluster {cluster_id} (No Title)")
    # Plotting
                     
                     

                    # plt.subplot(n_rows, n_cols, i + 1)
                     ax = axes[i] 
                     ax.imshow(wordcloud, interpolation='bilinear')
                     ax.set_title(cluster_title, fontsize=12, wrap=True)
                     ax.axis('off')
             
                    # plt.subplot(n_rows, n_cols, i + 1) # Create subplot
                    # plt.imshow(wordcloud, interpolation='bilinear')
                    # plt.title(cluster_title, fontsize=12, wrap=True) # Use the meaningful title
                    # plt.axis('off')
              for j in range(n_clusters, n_rows * n_cols):
                                axes[j].set_visible(False)
              plt.tight_layout() # Adjust layout to prevent titles/plots from overlapping
              plt.suptitle("Word Clouds for News Article Clusters", y=1.02, fontsize=16) # Overall title
              st.pyplot(fig)
              plt.close(fig)
              custom_query = st.sidebar.text_input(
                ":rainbow[Enter custom search keywords:]")
              if custom_query !="": 
                     pre_text=preprocess_text(custom_query)  

                     vect_text=vect.transform([pre_text])
                     predict=(kmean.predict(vect_text))
                     cluster = cluster_name_mapping.get(predict[0], f"Cluster {predict[0]} (No Title)")
                     st.sidebar.write(f"**_:rainbow[{cluster}]_**")
                     res=get_vader_sentiment(pre_text)
                    
                     result_string_join = "\n".join([f"{key}:{value}" for key, value in res.items()])
                     st.sidebar.text(result_string_join)
  #####################################################################################################################     
    elif selected_question == 'Report':
        st.header("📊 :rainbow[Yearly News Report]", divider="rainbow")

        # Assume you have a combined.csv or similar file with historical data
        # For a full yearly report, you'd need significantly more data than the GNews API free tier provides
        try:
            df_report = pd.read_csv(r'c:\users\91904\combinedsent.csv')
            df_report['Date'] = pd.to_datetime(df_report['date']) # Ensure 'Date' column is datetime
            df_report['Year'] = df_report['Date'].dt.year
            df_report['Month'] = df_report['Date'].dt.month

            available_years = sorted(df_report['Year'].unique())
            selected_year = st.sidebar.selectbox(":rainbow[Select a Year for Report:]", available_years, index=len(available_years)-1 if available_years else None)

            if selected_year:
                yearly_data = df_report[df_report['Year'] == selected_year].copy()
              
              

                if not yearly_data.empty:
                    # Apply sentiment analysis if not already present
                    if 'Overall_sentiment' not in yearly_data.columns:
                        sentiment_results_report = yearly_data['short_description'].apply(lambda x: pd.Series(get_vader_sentiment(x)))
                        yearly_data = pd.concat([yearly_data, sentiment_results_report], axis=1)

                    st.subheader(f":blue[Report for Year {selected_year}]")
                    st.write(f"Total articles analyzed: **{len(yearly_data)}**")
                    st.write("---")

                    # 1. Overall Sentiment Distribution
                    st.markdown("### :rainbow[Overall Sentiment Distribution] 📈")
                    sentiment_counts = yearly_data['Overall_sentiment'].value_counts()
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax1,
                                            colors=['lightgreen', 'lightcoral', 'lightskyblue'])
                    ax1.set_ylabel('') # Hide the default 'count' label
                    ax1.set_title(f'Overall Sentiment Distribution for {selected_year}')
                    st.pyplot(fig1)
                    plt.close(fig1)
                    st.write("---")
                    # 2. Sentiment Trend Over Months
                    st.markdown("### :rainbow[Monthly Sentiment Trend ]📊")
                    monthly_sentiment = yearly_data.groupby('Month')[['Positive', 'Negative', 'Neutral', 'Compound']].mean()
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    monthly_sentiment[['Positive', 'Negative', 'Neutral']].plot(kind='line', ax=ax2)
                    ax2.set_title(f'Average Sentiment Scores by Month in {selected_year}')
                    ax2.set_xlabel('Month')
                    ax2.set_ylabel('Average Sentiment Score')
                    ax2.set_xticks(range(1, 13))
                    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                    if len(monthly_sentiment)>1:
                          st.pyplot(fig2)
                    else:
                          st.write(monthly_sentiment)
                    plt.close(fig2)
                    st.write("---")
                     # 3. Top Categories by Volume
                    st.markdown("### :rainbow[Top News Categories by Volume] 🏆")
                    category_counts = yearly_data['category'].value_counts().head(10)
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    category_counts.plot(kind='barh', ax=ax3, color='yellow')
                    ax3.set_title(f'Top 10 News Categories in {selected_year}')
                    ax3.set_xlabel('Number of Articles')
                    ax3.set_ylabel('Category')
                    st.pyplot(fig3)
                    plt.close(fig3)
                    st.write("---")
                    # 4. Most Frequent Keywords Word Cloud
                    st.markdown("### :rainbow[Most Frequent Keywords/Topics ]📝")
                    all_text_for_wordcloud = " ".join(yearly_data['headline'].dropna())
                    if all_text_for_wordcloud:
                        wordcloud_report = WordCloud(width=1000, height=500, background_color='white',
                                                    stopwords=stop_words, max_words=100).generate(all_text_for_wordcloud)
                        fig4, ax4 = plt.subplots(figsize=(12, 8))
                        ax4.imshow(wordcloud_report, interpolation='bilinear')
                        ax4.axis('off')
                        ax4.set_title(f'Most Frequent Keywords in {selected_year} News')
                        st.pyplot(fig4)
                        plt.close(fig4)
                    else:
                        st.info("No text available to generate word cloud for this year.")
                    st.write("---")
                    # 5. Sentiment by Category
                    st.markdown("### :rainbow[Average Sentiment by Category ]🌈")
                    avg_sentiment_by_category = yearly_data.groupby('category')['Compound'].mean().sort_values(ascending=False).head(10)
                    fig5, ax5 = plt.subplots(figsize=(10, 6))
                    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    avg_sentiment_by_category.plot(kind='bar', ax=ax5, color=custom_colors)
                    ax5.set_title(f'Average Compound Sentiment by Category in {selected_year}')
                    ax5.set_xlabel('Category')
                    ax5.set_ylabel('Average Compound Sentiment Score')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig5)
                    plt.close(fig5)
                    st.write("---")

        except FileNotFoundError:
            st.error("Error: 'combinedsenti.csv' not found. Please make sure the data file is in the correct path or generate some news data first.")
        except Exception as e:
            st.error(f"An error occurred while generating the report: {e}")







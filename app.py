import streamlit as st
import numpy as np
import pandas as pd
import os
import chromadb
import requests
import seaborn as sns
import time
import plotly.express as px
from requests.adapters import HTTPAdapter
import logging
from typing import Optional
from requests.packages.urllib3.util.retry import Retry

from sentence_transformers import SentenceTransformer
import torch

headers = {'bypass-tunnel-reminder': 'any_value'}
response = requests.get('http://akabi.loca.lt', headers=headers)

## page settings
st.set_page_config(page_title="AmpleCart Feedback App", layout="wide")

# ChromaDB client
@st.cache_resource
def get_database_client():
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    heartbeat = chroma_client.heartbeat()
    if not heartbeat:
        st.error("Failed to connect to ChromaDB. Please check the logs for more details.")
        st.stop()
    collection = chroma_client.get_collection(name="SentimentAnalysis_3")

    if not collection:
        st.error("Failed to get the collection. Please check the logs for more details.")
        st.stop()

    return collection

# Set the title of the page
st.title('AmpleCart feedback analysis app')

# Add some text to describe your app
st.sidebar.write('This app queries a subset of reviews using similarity search.')

# Placeholder for future database query results and visualizations
#st.write('Database query results and visualizations will appear here.')

#api_token = st.text_input('Enter your API token here)', 'API token')
# Load the Excel file with the topic mappings
topic_mapping_path = 'Topic Categories Mapping.xlsx'
topic_mappings = pd.read_excel(topic_mapping_path)


# Get the categorys from the topic mappings
categories = topic_mappings['Category'].unique()

# Example query


# Create a text input widget for entering the query
query_text = st.sidebar.text_input('Enter your query search here:', 'The shipping was') 

knum = st.sidebar.slider("Select amount of query results", 100, 4000, value=500, step=100)
selected_categories =st.sidebar.multiselect("Select one, many or no Categories",categories)


# Create a 'where' clause for the query based on the selected categories
if len(selected_categories) == 1:
    where_clause = {'category': {'$eq': selected_categories[0]}}
elif len(selected_categories) > 1:
    where_clause = {'category': {'$in': selected_categories}}
else:
    # If no categories are selected, you can choose to omit the 'where' clause or tailor it as needed
    where_clause = {}


@st.cache_resource
def model_load():
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)
# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# use a sentence transformer model to encode the query text
retriver = model_load()

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)


query_embedding = retriver.encode(query_text).tolist()

# API token
#api_token = 

#API_TOKEN = api_token

#embedding_cache = {}
#API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
#headers = {"Authorization": f"Bearer {API_TOKEN}"}
#def query(payload):
#    response = requests.post(API_URL, headers=headers, json=payload)
#    return response.json()
#query_embedding = query(query_text)


# Show and update progress bar
collection = get_database_client()
# query embedding function
text_query = collection.query(
    query_embeddings=query_embedding,
    include=['documents', 'metadatas', 'distances'],
    n_results=knum,
    where=where_clause
)


for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()


rows = []
# Assuming each key's first list item contains lists of equal length
for i in range(len(text_query['ids'][0])):
    row = {
        'id': text_query['ids'][0][i],
        'document': text_query['documents'][0][i]
    }
    # Add metadata fields to the dictionary, checking for metadata presence
    if text_query['metadatas']:
        row.update(text_query['metadatas'][0][i])

    # Add distance to the dictionary, with proper indentation
    if text_query['distances']:
        row['distance'] = text_query['distances'][0][i]
    
    rows.append(row)

# Create a DataFrame from the list of dictionaries
@st.cache_data
def get_dataframe():
    return pd.DataFrame(rows)
df = get_dataframe()


# Sidebar for page selection using radio buttons
#page = st.radio('Choose a page:', ['Dataframe', 'Visuals'])

tab1, tab2 = st.tabs(["Dataframe", "Visuals"])

topic_mappings['Category'].unique()


## visualizations code

df['sentiment'] = df['sentiment'].astype(int)

# Add sentiment counts, making sure all values from 1 to 5 are present
sentiment_counts = df['sentiment'].value_counts().reindex(range(1, 6), fill_value=0)

# Reset the index to turn it into a DataFrame
data = sentiment_counts.reset_index()
data.columns = ['sentiment', 'count']

# Add star emojis to sentiment labels
data['sentiment'] = data['sentiment'].apply(lambda x: f"{x} ⭐️")

# Filter data based on selected categories
filtered_data = df[df['category'].isin(selected_categories)]

# Limit to top 3 categories if more than 3 are selected
if len(selected_categories) > 5:
    top_categories = filtered_data['category'].value_counts().index[:5]
    filtered_data = filtered_data[filtered_data['category'].isin(top_categories)]
    selected_categories = top_categories.tolist()  # Updating the selected categories to the top 3

# Aggregate data for overall sentiment counts (ignoring categories)
overall_sentiment_counts = df.groupby('sentiment').size().reset_index(name='count')
overall_sentiment_counts['sentiment'] = overall_sentiment_counts['sentiment'].apply(lambda x: f"{x} ★")

# mean sentiment score for each category and overall
mean_sentiment = df.groupby('category')['sentiment'].mean().reset_index(name='mean_sentiment')
overall_mean_sentiment = df['sentiment'].mean()
# filtered mean sentiment score for each category
filtered_data = df[df['category'].isin(selected_categories)]
filtered_mean_sentiment = filtered_data.groupby('category')['sentiment'].mean().reset_index(name='mean_sentiment')
#filtered_mean_sentiment['mean_sentiment'] = filtered_mean_sentiment['mean_sentiment'].apply(lambda x: f"{x:.1f} ★")

# Average sentiment score set by the whole dataset
avg_sentiment_score_wholeset = 3.20

# Function to get delta score 
def get_delta_score_overall(avg_sentiment_score_wholeset, overall_mean_sentiment):
    delta_score = overall_mean_sentiment - avg_sentiment_score_wholeset
    return delta_score

def get_delta_score_category(overall_mean_sentiment, category_mean_sentiment):
    delta_score = category_mean_sentiment - overall_mean_sentiment
    if delta_score < 0:
        # Negative change, should be red and downward arrow
        return f"{delta_score:.2f}", "normal"  # Assuming 'negative' turns text red
    else:
        # Positive change, should be green and upward arrow
        return f"{delta_score:.2f}", "normal"  # Assuming 'positive' turns text green



# Create a dataframe for the ag grid
df_dataframe = df[['document', 'category', 'sentiment']]

# add star to the sentiment
df_dataframe['sentiment'] = df_dataframe['sentiment'].apply(lambda x: f"{x} ★")

# Display the selected page

with tab1:
    col1, col2 = st.columns([2,1],gap="small")
    col1.title('A DataFrame of the query results')
    col1.dataframe(df_dataframe,use_container_width=True,height=500, hide_index=True, 
                   column_config={'document': {'width': 500}})
    # Title for the plot
    # Title for the plot
    col2.title('Top 5 Categories by Count')

    # Calculate the top 5 categories
    top_topics = df['category'].value_counts().head(5).reset_index()
    top_topics.columns = ['Category', 'Count']
    top_topics = top_topics.sort_values('Count', ascending=True)

    # Create a horizontal bar chart with Plotly Express
    
    fig = px.bar(top_topics, y='Category', x='Count', orientation='h')

    # Place the category labels after the bars
    fig.update_traces(textposition='outside')

    # Display the plot in the Streamlit app
    col2.plotly_chart(fig)

with tab2:
    st.title('Visuals Page')
    
    cols = st.columns(len(selected_categories)+1)
    with cols[0]:  # This will display the overall sentiment score in the first column
        st.metric(label="Sentiment Score", 
                  value=f"{overall_mean_sentiment:.2f} ★", 
                  delta=f"{get_delta_score_overall(avg_sentiment_score_wholeset, overall_mean_sentiment):.2f} ★",
                  help="The average sentiment score for the whole dataset is 3.20 ★")

    if not selected_categories:  # This should be fine if selected_categories is a list
        # if no category is selected, display nothing
        st.empty()
    else:
        for i, category in enumerate(selected_categories, start=1):  # Start from 1 to leave the first column for the overall score
            with cols[i]:  # This will place each category score in its own column
                category_mean_sentiment = filtered_mean_sentiment[filtered_mean_sentiment['category'] == category]['mean_sentiment'].iloc[0]
                delta, delta_color = get_delta_score_category(overall_mean_sentiment, category_mean_sentiment)
                st.metric(
                    label=f"{category} Sentiment Score", 
                    value=f"{category_mean_sentiment:.2f} ★", 
                    delta=delta,
                    delta_color=delta_color)
    col1, col2 = st.columns(2)
    # Sentiment counts plot (overall, not filtered by category)

    sentiment_fig = px.bar(overall_sentiment_counts, x='sentiment', y='count',
                               title='Overall Sentiment Counts',
                               labels={'sentiment': 'Sentiment', 'count': 'Count'})
    col1.plotly_chart(sentiment_fig, use_container_width=True)
        
    # Distribution plots for each selected category
    if not selected_categories:  # This should be fine if selected_categories is a list
        col2.write('No category selected. Please select a category to view this chart.')
    else:
        category_sentiment_counts = filtered_data.groupby(['category', 'sentiment']).size().reset_index(name='count')
        category_sentiment_counts['sentiment'] = category_sentiment_counts['sentiment'].apply(lambda x: f"{x} ★")
    
        sentiment_distribution_fig = px.bar(category_sentiment_counts, x='sentiment', y='count',
                                    color='category', barmode='group',
                                    title='Sentiment Distribution by Category (Top 5)',
                                    labels={'count': 'Count', 'sentiment': 'Sentiment', 'category': 'Category'})
        col2.plotly_chart(sentiment_distribution_fig, use_container_width=True)
        
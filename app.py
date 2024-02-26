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

## page settings
st.set_page_config(page_title="AmpleCart Feedback App", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)
# ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
heartbeat = chroma_client.heartbeat()
if not heartbeat:
    st.error("Failed to connect to ChromaDB. Please check the logs for more details.")
    st.stop()
collection = chroma_client.get_collection(name=
        "SentimentAnalysis_3")
collection_count = collection.count()
if not collection:
    st.error("Failed to get the collection. Please check the logs for more details.")
    st.stop()

st.sidebar.write("Collection count:", collection_count)
# Set the title of the page
st.sidebar.title('AmpleCart feedback analysis app')

# Add some text to describe your app
st.sidebar.write('This app queries a subset of reviews using similarity search.')

# Placeholder for future database query results and visualizations
#st.write('Database query results and visualizations will appear here.')

#api_token = st.text_input('Enter your API token here)', 'API token')
api_token = "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"

# Example query

# Create a text input widget for entering the query
query_text = st.sidebar.text_input('Enter your query here:', 'The shipping was') 

# Create a slider for selecting the number of query results to return
knum = st.sidebar.slider("Select amount of query results", 1, 2000, value=10, step=50)

# Load the Excel file with the topic mappings
topic_mapping_path = 'Topic Categories Mapping.xlsx'
topic_mappings = pd.read_excel(topic_mapping_path)

# Get the categorys from the topic mappings
categories = topic_mappings['Category'].unique()

# Create a multiselect widget for selecting categories
selected_categories =st.sidebar.multiselect("Select one, many or no Categories",categories)

# Create a 'where' clause for the query based on the selected categories
if len(selected_categories) == 1:
    where_clause = {'category': {'$eq': selected_categories[0]}}
elif len(selected_categories) > 1:
    where_clause = {'category': {'$in': selected_categories}}
else:
    # If no categories are selected, you can choose to omit the 'where' clause or tailor it as needed
    where_clause = {}

# API token
api_token = "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"

API_TOKEN = api_token

embedding_cache = {}
API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
query_embedding = query(query_text)


#st.write("Checkpoint: After embedding")
#st.write(embedding)

# query embedding function
text_query = collection.query(
    query_embeddings=query_embedding,
    include=['documents', 'metadatas', 'distances'],
    n_results=knum,
    where=where_clause
)


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
df = pd.DataFrame(rows)

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

# Display the selected page

with tab1:
    st.spinner('Loading query results...')
    st.title('A DataFrame of the query results')
    st.dataframe(df, use_container_width=True)

with tab2:
    st.spinner('Loading visualizations...')
    st.title('Visuals Page')
    col1, col2 = st.columns(2)
        
    # Sentiment counts plot (overall, not filtered by category)
    col1.write('Overall Sentiment Counts')
    sentiment_fig = px.bar(overall_sentiment_counts, x='sentiment', y='count',
                               title='Overall Sentiment Counts',
                               labels={'sentiment': 'Sentiment', 'count': 'Count'})
    col1.plotly_chart(sentiment_fig, use_container_width=True)
        
    # Distribution plots for each selected category
    col2.write('Sentiment Distribution by Category (Top 5)')
    if not selected_categories:  # This should be fine if selected_categories is a list
        col2.write('No category selected. Please select a category to view this chart.')
    else:
        filtered_data = df[df['category'].isin(selected_categories)]

        category_sentiment_counts = filtered_data.groupby(['category', 'sentiment']).size().reset_index(name='count')
         # Ensure sentiments are labeled with stars in the plot
        category_sentiment_counts['sentiment'] = category_sentiment_counts['sentiment'].apply(lambda x: f"{x} ★")
    
        sentiment_distribution_fig = px.bar(category_sentiment_counts, x='sentiment', y='count',
                                    color='category', barmode='group',
                                    title='Sentiment Distribution by Category',
                                    labels={'count': 'Count', 'sentiment': 'Sentiment', 'category': 'Category'})
        col2.plotly_chart(sentiment_distribution_fig, use_container_width=True)
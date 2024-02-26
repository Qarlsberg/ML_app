import streamlit as st
import numpy as np
import pandas as pd
import os
import chromadb
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import time


chroma_client = chromadb.PersistentClient(path="chromadb/to")
heartbeat = chroma_client.heartbeat()
collection = chroma_client.get_collection(name=
        "SentimentAnalysis_3")

# Set the title of the page
st.title('AmpleCart feedback analysis app')

# Add some text to describe your app
st.write('This app queries a subset of reviews using similarity search.')

# Placeholder for future database query results and visualizations
#st.write('Database query results and visualizations will appear here.')

#api_token = st.text_input('Enter your API token here)', 'API token')
api_token = "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"

# Example query

query_text = st.sidebar.text_input('Enter your query here:', 'I am happy') 


knum = st.sidebar.slider("Select amount of query results", 1, 2000, value=10, step=50)

# Load the Excel file with the topic mappings
topic_mapping_path = 'Topic Categories Mapping.xlsx'
topic_mappings = pd.read_excel(topic_mapping_path)

# Get the categorys from the topic mappings
#categories = topic_mappings['Category'].unique()

#selected_categories =st.radio("Categories",categories)

#if len(selected_categories) == 0:
#    selected_categories = categories


API_TOKEN= api_token

API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-small-en-v1.5"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
with st.spinner('Wait for it...'):
    query_embedding = query(query_text)
    time.sleep(2)
    st.success('Done!')


#st.write("Checkpoint: After embedding")
#st.write(embedding)

# query embedding function
text_query = collection.query(
    query_embeddings=query_embedding,
    include=['documents', 'metadatas'],
    n_results=knum
    #where={'category': {'$eq': selected_categories}}
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
    
    rows.append(row)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(rows)


# Create a dictionary for mapping topics to categories from the Excel mapping data
topic_to_category_dict = topic_mappings.set_index('Topic')['Category'].to_dict()

# Create a new column in the dataframe with the category for each topic
df['category'] = df['topic'].map(topic_to_category_dict)



st.dataframe(df)

bar_chart = df['sentiment'].value_counts()
st.bar_chart(bar_chart)

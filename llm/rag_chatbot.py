import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
import pandas as pd

load_dotenv()

# Resolve paths relative to the project root (parent of llm/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHROMA_DB_PATH = os.path.join(_PROJECT_ROOT, 'chroma_db')
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

openai_emb_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name='text-embedding-3-small'
)

chroma_client = chromadb.PersistentClient(_CHROMA_DB_PATH)

collections = chroma_client.get_or_create_collection(
    name='nusantara_connect',
    metadata={'description': 'Nusantara Connect Profile & Customer Data'},
    embedding_function=openai_emb_func
)

def load_docs(folder_path):
    docs = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                docs.append({
                    'filename':file_name,
                    'content': content
                })
                
    return docs


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
            
        start += chunk_size - overlap
        
    return chunks

def add_docs_to_db(folder_path):
    docs = load_docs(folder_path)
    
    all_chunks = []
    all_ids = []
    all_metadatas = []
    
    chunk_counter = 0
    
    for doc in docs:
        chunks = chunk_text(doc['content'])
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f'chunk_ke_{chunk_counter}')
            all_metadatas.append({
                'source': doc['filename'],
                'chunk_id': i
            })
            
            chunk_counter += 1
            
    collections.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)

def add_csv_to_db(file_path, batch_size=100):
    df = pd.read_csv(file_path)
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
    
    df = pd.read_csv(file_path)
    batch_chunks = []
    batch_ids = []
    batch_metadatas = []

    for i, row in df.iterrows():
        content = f"Pelanggan ID {row['id']}, Gender: {row['gender']}, " \
                  f"Layanan: {row['InternetService']}, Kontrak: {row['Contract']}, " \
                  f"Monthly Charges: {row['MonthlyCharges']}, Status Churn: {row['Churn']}"
        
        batch_chunks.append(content)
        batch_ids.append(f"csv_{i}")
        batch_metadatas.append({"source": os.path.basename(file_path), "type": "customer_data"})

    collections.add(documents=batch_chunks, ids=batch_ids, metadatas=batch_metadatas)
    print(f"Berhasil menambahkan batch CSV: {i} sampai {i + len(batch_df)}")

def search(query, n_result=3, relevance_threshold=1.3):
    """Search the knowledge base with relevance filtering.
    
    Args:
        query: Search query text.
        n_result: Number of results to return.
        relevance_threshold: Maximum distance for a chunk to be considered
            relevant. Chunks with distance > threshold are excluded.
            Lower = stricter filtering. Typical range: 0.8 (strict) to 1.5 (loose).
            Default 1.3 provides good balance between precision and recall.
    """
    result = collections.query(
        query_texts=[query],
        n_results=n_result
    )
    
    relevant_chunks = []
    for i in range(len(result['documents'][0])):
        distance = result['distances'][0][i]
        # Filter out chunks that are too distant (not relevant)
        if distance <= relevance_threshold:
            relevant_chunks.append({
                'text': result['documents'][0][i],
                'source': result['metadatas'][0][i]['source'],
                'distance': distance
            })
        
    return relevant_chunks


def check_relevance(query, threshold=1.3):
    """Check if a query is relevant to the knowledge base.
    
    Returns True if at least one chunk is within the relevance threshold.
    Useful for pre-filtering out-of-topic questions.
    """
    results = search(query, n_result=1, relevance_threshold=threshold)
    return len(results) > 0

def generate_answer(history):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=history
    )
    
    return response.choices[0].message.content

def get_embedding(text):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vector1, vector2):
    """Fungsi ini digunakan untuk menghitung cosine similarity"""
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity

if collections.count() == 0:
    print('Database kosong, menambahkan dokumen')
    add_docs_to_db(os.path.join(_DATA_DIR, 'nusantara_connect'))
    add_csv_to_db(os.path.join(_DATA_DIR, 'train.csv'))
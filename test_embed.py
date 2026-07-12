import os
from config import GOOGLE_API_KEYS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    print("Testing models/embedding-001...")
    embeddings1 = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEYS[0])
    res1 = embeddings1.embed_documents(["Hello"])
    print("Success models/embedding-001")
except Exception as e:
    print("Error models/embedding-001:", e)

try:
    print("Testing models/text-embedding-004...")
    embeddings2 = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEYS[0])
    res2 = embeddings2.embed_documents(["Hello"])
    print("Success models/text-embedding-004")
except Exception as e:
    print("Error models/text-embedding-004:", e)

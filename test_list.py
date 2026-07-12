import os
import google.generativeai as genai
from config import GOOGLE_API_KEYS

try:
    genai.configure(api_key=GOOGLE_API_KEYS[0])
    print("Available models supporting embedContent:")
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print("Error:", e)

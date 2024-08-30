# ECHAT
Developed a RAG-based customized chatbot using Streamlit, Chroma, and FastEmbedEmbeddings for interactive
Excel file querying.

## Create Virtual Environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Installation 
pip install streamlit chromadb fastembed

## Ollama
pip install ollama 

ollama pull model-name

ollama run model-name

## Running App
streamlit run app.py

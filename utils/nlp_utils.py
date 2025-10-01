import nltk
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer 
import string
from typing import Optional

nltk_data_dir = os.path.join("/tmp", "nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print("NLTK 'punkt' e 'stopwords' já carregados.")

except LookupError:
    print("Baixando NLTK 'punkt' e 'stopwords' para /tmp/nltk_data...")
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    
except Exception as e:
    print(f"Erro na inicialização do NLTK: {e}")

try:
    STOP_WORDS = set(stopwords.words('portuguese'))
except Exception:
    STOP_WORDS = set() 

WORD_PUNCT_TOKENIZER = WordPunctTokenizer()
    
def preprocess_text(text: Optional[str]) -> str:
    """
    Limpa e normaliza o texto, removendo URLs, pontuação e stop words.
    Ideal para modelos de ML tradicionais, mas não usado diretamente 
    no prompt do LLM para preservar o contexto.
    """
    if not text or not isinstance(text, str):
        return ""

  
    text = text.lower()

   
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

  
    tokens = WORD_PUNCT_TOKENIZER.tokenize(text)
    
    punctuations = set(string.punctuation)

    filtered_tokens = []
    for word in tokens:
    
        if word in STOP_WORDS:
            continue
        if word in punctuations:
            continue
    
        if word.isdigit() or len(word) < 2:
            continue
       
        clean_word = re.sub(r'[^a-z0-9]', '', word)
       
        if clean_word and clean_word not in STOP_WORDS and len(clean_word) >= 2:
            filtered_tokens.append(clean_word)
   
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text.strip()
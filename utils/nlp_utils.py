import nltk
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer # <<< NOVA IMPORTAÇÃO
import string
from typing import Optional

# --- 1. Configuração do NLTK para o Vercel/Lambda ---
# Define o diretório /tmp para o download dos dados do NLTK.
nltk_data_dir = os.path.join("/tmp", "nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Baixa os recursos necessários (stopwords é essencial, punkt é só por garantia)
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

# Inicializa as ferramentas APÓS garantir que foram baixadas
try:
    STOP_WORDS = set(stopwords.words('portuguese'))
except Exception:
    STOP_WORDS = set() 

# Inicializa o tokenizador WordPunct uma única vez
WORD_PUNCT_TOKENIZER = WordPunctTokenizer()
# --- Fim da Configuração do NLTK ---
    
def preprocess_text(text: Optional[str]) -> str:
    """
    Realiza o pré-processamento robusto do texto usando WordPunctTokenizer
    para evitar a dependência de 'punkt_tab'.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Minúsculas
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Tokeniza usando WordPunctTokenizer (não requer arquivos de dados externos)
    tokens = WORD_PUNCT_TOKENIZER.tokenize(text)
    
    # Define pontuação a ser removida (apenas caracteres únicos)
    punctuations = set(string.punctuation)

    # 4. Remove Stopwords, pontuação e tokens inválidos
    filtered_tokens = []
    for word in tokens:
        # Pula se for stopword
        if word in STOP_WORDS:
            continue
        # Pula se for pontuação (WordPunctTokenizer separa pontuação)
        if word in punctuations:
            continue
        # Pula se for puramente numérico ou token de caractere único
        if word.isdigit() or len(word) < 2:
            continue
        
        # O WordPunctTokenizer tende a ser mais sujo, então limpamos aqui
        # Remove caracteres que podem ter sobrado (ex: traços, apóstrofos)
        clean_word = re.sub(r'[^a-z0-9]', '', word)
        
        if clean_word and clean_word not in STOP_WORDS and len(clean_word) >= 2:
            filtered_tokens.append(clean_word)
    
    # 5. Junta o texto novamente
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text.strip()
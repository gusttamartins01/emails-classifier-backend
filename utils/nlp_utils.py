import nltk
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. Configuração do NLTK para o Vercel/Lambda ---

# Define o diretório /tmp para o download dos dados do NLTK,
# pois o ambiente Vercel é somente leitura exceto em /tmp.
nltk_data_dir = os.path.join("/tmp", "nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Baixa os recursos necessários, se não estiverem presentes
try:
    # Tenta carregar o 'punkt' para verificar se já existe
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print("NLTK 'punkt' e 'stopwords' já carregados.")

except LookupError:
    # Se não encontrar (LookupError), faz o download para o diretório /tmp
    print("Baixando NLTK 'punkt' e 'stopwords' para /tmp/nltk_data...")
    
    # É mais robusto e seguro usar nltk.download() com o download_dir
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)

# --- Fim da Configuração do NLTK ---


# Inicializa as stopwords APÓS garantir que foram baixadas
try:
    STOP_WORDS = set(stopwords.words('portuguese'))
except Exception as e:
    # Caso as stopwords ainda falhem, usa um conjunto vazio
    print(f"Erro ao carregar stopwords: {e}")
    STOP_WORDS = set() 
    
def preprocess_text(text: str) -> str:
    """
    Realiza o pré-processamento básico do texto:
    - Minúsculas
    - Remoção de caracteres não alfanuméricos (exceto espaços)
    - Tokenização e remoção de stopwords
    """
    if not text:
        return ""

    # 1. Minúsculas
    text = text.lower()
    
    # 2. Remove URLs e pontuação
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Tokeniza
    tokens = word_tokenize(text, language='portuguese')
    
    # 4. Remove Stopwords
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    
    # 5. Junta o texto novamente
    processed_text = ' '.join(filtered_tokens)
    
    return processed_text.strip()
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer # <-- Nova importação
import string
from typing import Optional

# ----------------------------------------------------
# 1. GARANTIA DE DOWNLOAD NLTK (ESSENCIAL PARA O VERCEL)
# Manteremos os downloads de 'punkt' e 'stopwords' para garantir os recursos.
# O 'WordPunctTokenizer' não precisa de 'punkt', mas a lógica de 'sent_tokenize'
# dentro do NLTK pode precisar dele, e as stopwords são essenciais.
# ----------------------------------------------------
try:
    # Tenta verificar se os recursos já existem
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    # Se não existirem, faz o download
    nltk.download('punkt')
    nltk.download('stopwords')

# Tenta carregar os recursos após o download/verificação
try:
    stop_words = set(stopwords.words('portuguese'))
except LookupError:
    stop_words = set()
    
# Inicializa o tokenizador WordPunct uma única vez
word_tokenizer = WordPunctTokenizer()
# ----------------------------------------------------


def preprocess_text(texto: Optional[str]) -> str:
    """
    Realiza o pré-processamento de texto usando NLTK (leve e robusto).
    Utiliza WordPunctTokenizer para evitar falhas de recursos como 'punkt_tab'.
    """
    
    # 1. Validação de entrada
    if not texto or not isinstance(texto, str):
        return ""

    # Inicializa a lista de tokens processados
    processados = []

    # 2. Converte para minúsculas e remove espaços extras
    texto_limpo = texto.lower().strip()

    # 3. Tokenização
    # Agora usamos o WordPunctTokenizer.tokenize() para evitar a dependência de punkt_tab.
    tokens = word_tokenizer.tokenize(texto_limpo)

    # Define pontuação a ser removida
    punctuations = set(string.punctuation)

    # 4. Pré-processamento (filtragem)
    for token in tokens:
        # Pula se for stopword, pontuação, ou número
        if token in stop_words:
            continue
        if token in punctuations:
            continue
        if token.isdigit():
            continue
        
        processados.append(token)

    # 5. Junta os tokens de volta em uma string
    return " ".join(processados)
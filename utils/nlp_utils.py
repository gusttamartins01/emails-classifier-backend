import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from typing import Optional

# ----------------------------------------------------
# 1. GARANTIA DE DOWNLOAD NLTK (ESSENCIAL PARA O VERCEL)
# Este bloco garante que os recursos do NLTK estejam disponíveis.
# Ele roda na inicialização e é a chave para o Vercel não falhar por LookupError.
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
    # Carrega as stopwords do idioma português
    stop_words = set(stopwords.words('portuguese'))
except LookupError:
    # Em caso de falha de carregamento (improvável), usa um set vazio
    stop_words = set()
# ----------------------------------------------------


def preprocess_text(texto: Optional[str]) -> str:
    """
    Realiza o pré-processamento de texto usando NLTK (leve).
    Inclui conversão para minúsculas, tokenização e remoção de pontuação e stopwords.
    """
    
    # 1. Validação de entrada
    if not texto or not isinstance(texto, str):
        return ""

    # Inicializa a lista de tokens processados
    processados = []

    # 2. Converte para minúsculas e remove espaços extras
    texto_limpo = texto.lower().strip()

    # 3. Tokenização
    # **IMPORTANTE:** Removemos 'language='portuguese'' para evitar o erro 'punkt_tab'.
    # O tokenizador padrão é suficiente para a maioria das tarefas de limpeza.
    tokens = word_tokenize(texto_limpo)

    # Define pontuação a ser removida
    punctuations = set(string.punctuation)

    # 4. Pré-processamento (filtragem)
    for token in tokens:
        # Pula se for stopword ou pontuação
        if token in stop_words:
            continue
        if token in punctuations:
            continue
            
        # Pula se for um token puramente numérico (ex: '2025')
        if token.isdigit():
            continue
        
        processados.append(token)

    # 5. Junta os tokens de volta em uma string
    return " ".join(processados)
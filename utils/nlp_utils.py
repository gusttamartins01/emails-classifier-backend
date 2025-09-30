import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Baixa os recursos necessários do NLTK.
# OBS: O NLTK deve baixar estes arquivos uma vez durante o build do Vercel.
# Para o ambiente local, você pode precisar executá-los manualmente uma vez:
# nltk.download('punkt')
# nltk.download('stopwords')

# Define os recursos
try:
    # Tenta carregar os recursos
    stop_words = set(stopwords.words('portuguese'))
except LookupError:
    # Se falhar, baixa-os (isso deve acontecer no build do Vercel)
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('portuguese'))


def preprocess_text(texto: str) -> str:
    if not texto or not isinstance(texto, str):
        return ""

    # 1. Converte para minúsculas e remove espaços extras
    texto = texto.lower().strip()

    # 2. Tokenização
    tokens = word_tokenize(texto, language='portuguese')

    # 3. Pré-processamento (removendo pontuação, stopwords e números)
    processados = []
    
    # Define pontuação a ser removida (string.punctuation)
    punctuations = set(string.punctuation)
    
    for token in tokens:
        # Se for um token de pontuação, ignora
        if token in punctuations:
            continue
        
        # Se for uma stopword, ignora
        if token in stop_words:
            continue
            
        # Remove tokens que são puramente números (ex: '2025')
        if token.isdigit():
            continue

        # Adiciona o token (o NLTK não faz lematização simples como o spacy,
        # mas para filtros de e-mail, esta versão é muito mais leve e suficiente)
        processados.append(token)

    # 4. Junta os tokens de volta em uma string
    return " ".join(processados)

# Exemplo de uso (opcional, remova antes do push final se não precisar)
# print(preprocess_text("O email de SPAM chegou na quarta-feira 23!"))
# print(preprocess_text("Este é um e-mail de teste importante."))
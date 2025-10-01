import os
import json
import asyncio
import PyPDF2
from dotenv import load_dotenv
from io import BytesIO 

# Certifique-se de que utils/groq_client.py e utils/nlp_utils.py existem
from utils.groq_client import gerar_resposta
from utils.nlp_utils import preprocess_text

# Carrega a chave GROQ_API_KEY do arquivo .env
load_dotenv() 

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Configuração do CORS (necessário para o frontend rodando em outro domínio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite qualquer origem (para desenvolvimento/Vercel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    """Endpoint de saúde/teste da API."""
    return {"message": "API funcionando 🚀"}

@app.post("/process")
async def process_email(text: str = Form(None), file: UploadFile = File(None)):
    """
    Processa um texto ou um arquivo PDF para classificação e geração de resposta pela IA.
    """
    if not text and not file:
        raise HTTPException(status_code=400, detail="Nenhum texto ou arquivo PDF enviado.")

    # 1. Processamento de Arquivo PDF (Se um arquivo foi enviado)
    if file:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos.")

        try:
            content = await file.read()
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))

            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text() or ""

            text = pdf_text.strip()
            if not text:
                raise HTTPException(status_code=400, detail="Não foi possível extrair texto do PDF.")
        except Exception as e:
            print(f"[ERRO] Falha ao ler PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erro ao ler PDF: {str(e)}")

    try:
        
        texto_limpo = preprocess_text(text) 
        
        resultado = await gerar_resposta(text)

        if "Erro" in resultado.get("categoria", ""):
          
            return JSONResponse(content=resultado, status_code=503) 

     
        return JSONResponse(content=resultado)

    except Exception as e:
       
        erro_msg = str(e)
        print(f"[ERRO] Falha crítica ao gerar resposta: {erro_msg}")
        import traceback; traceback.print_exc()
        
        return JSONResponse(content={
            "categoria": "Erro Interno do Servidor",
            "resposta": "Ocorreu um erro inesperado ao processar o email. (Verifique os logs do servidor para detalhes.)"
        }, status_code=500)
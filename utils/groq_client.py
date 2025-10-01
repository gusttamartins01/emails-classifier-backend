import os
import json
from dotenv import load_dotenv
from groq import Groq, APIError 
import asyncio 

load_dotenv()
try:

    client = Groq()
except Exception:
 
    client = None

async def gerar_resposta(texto: str):
    
    if not os.getenv("GROQ_API_KEY") or client is None:
        return {
            "categoria": "Erro de Configuração",
            "resposta": "A chave da API (Groq) não está configurada ou o cliente falhou ao iniciar."
        }

    MODEL = "llama-3.1-8b-instant" 

    try:
    
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente profissional que classifica emails como 'Produtivo' ou 'Improdutivo'. "
                        
                        "**Regras de Classificação:** 1. Classifique como 'Produtivo' se o email contiver dúvidas, pedidos de ajuda, solicitações de suporte técnico, orientações sobre uso de aplicativos, ou qualquer demanda voltada ao trabalho, tecnologia ou assistência prática. 2. Classifique como 'Produtivo' também se o email demonstrar necessidade de acolhimento. 3. Classifique como 'Improdutivo' se o email for saudações genéricas, spam, ou assuntos pessoais irrelevantes. "
                        
                        "Gere uma resposta profissional, empática e acessível. "
                        "**Regras de Resposta:** Sempre retorne APENAS um objeto JSON com os campos 'categoria' e 'resposta'. NUNCA inclua texto adicional ou formatação Markdown. "
                        "Exemplo: {'categoria': 'Produtivo', 'resposta': '...'}"
                    )
                },
                {
                    "role": "user",
                    "content": "Classifique o seguinte email: " + texto
                }
            ],
            temperature=0.0
        )
        
        resposta_raw = response.choices[0].message.content.strip()

        try:
            resposta_json = json.loads(resposta_raw) 
            
            if isinstance(resposta_json, dict) and "categoria" in resposta_json and "resposta" in resposta_json:
                return resposta_json
            else:
                return {
                    "categoria": "Classificado pela IA (JSON Inválido)",
                    "resposta": "O LLM retornou um JSON com formato inesperado. Texto original: " + resposta_raw
                }

        except json.JSONDecodeError:
            return {
                "categoria": "Classificado pela IA (Erro de Parsing)",
                "resposta": "O LLM retornou um texto que não é um JSON válido. Texto original: " + resposta_raw
            }

    except APIError as e:
     
        api_error_message = getattr(e.response, 'text', str(e.body)) if hasattr(e, 'response') and e.response else str(e)
        print(f"[ERRO] Erro da API da Groq: {e}")
        return {
            "categoria": "Erro da API",
            "resposta": f"A API da Groq retornou um erro: {getattr(e, 'status_code', 'N/A')} - {api_error_message}"
        }
    except Exception as e:
       
        print(f"[ERRO] Falha geral: {e}")
        return {
            "categoria": "Erro de Conexão",
            "resposta": "Ocorreu um erro inesperado ao conectar ou processar a requisição."
        }
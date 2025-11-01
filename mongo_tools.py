import os
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Optional, List
from langchain.tools import tool
from pydantic import BaseModel, Field
import requests

load_dotenv()

model_name = "paraphrase-multilingual-MiniLM-L12-v2"
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
API_URL = f"https://router.huggingface.co/hf-inference/models/sentence-transformers/{model_name}/pipeline/feature-extraction"
headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}

# Embedding
def gerar_texto_embedding_receita(nome_receita, ingrediente, descricao, modo_preparo):
    partes = []

    # nome
    if nome_receita:
        partes.append(f"Nome: {nome_receita}")

    if descricao:
        partes.append(f"Descrição: {descricao}")

    if ingrediente:
        partes.append(f"Ingrediente(s): {ingrediente}")

    if modo_preparo:
        partes.append(f"Modo de preparo: {modo_preparo}")

    # montar texto final
    texto = "\n".join(partes).strip()

    return texto or None  # retorna None se nada aproveitável existir

def embed_text_api(texts):
    response = requests.post(API_URL, headers=headers, json={"inputs": texts, "options":{"wait_for_model": True}}) #wait_for_model espera pelo modelo caso ele esteja sobrecarregado, ao invés de retornar um erro

    print(f"{response.status_code} - {response.reason}")
    
    if response.status_code != 200:
        print(response.text)

    return response.json() 

MONGO_URL = os.getenv("MONGO_URL")

def get_collection():
    client = MongoClient(MONGO_URL)
    db = client["dbCodCoz"]
    coll = db["receitas"]
    return coll

class QueryReceitasModel (BaseModel):
    nome_receita: Optional[str] = Field(default=None, description="Nome da receita a ser pesquisada.")
    ingrediente: Optional[str] = Field(default=None, description="Ingredientes a serem identificados na pergunta. Se identificar mais de um, colocar nesse exemplo: 'alface, arroz'")
    descricao: Optional[str] = Field(default=None, description="Descrição básica ou apresentação da receita.")
    modo_preparo: Optional[str] = Field(default=None, description="Detalhes sobre o modo de preparo.")
    empresa_id: int = Field(..., description="ID da empresa para filtrar as receitas.")

@tool("query_receitas", args_schema=QueryReceitasModel)
def query_receitas(
    nome_receita: Optional[str] = None,
    ingrediente: Optional[str] = None,
    descricao: Optional[str] =  None,
    modo_preparo: Optional[str] = None,
    empresa_id: int = None
) -> list[dict]:
    """
    Consulta receitas na collection 'receitas' do MongoDB com filtro obrigatório de empresa_id e os opcionais: nome da receita, ingrediente(s), descrição mínima ou detalhes sobre o modo de preparo.
    """

    coll = get_collection()
    query = []

    texto_embedding = gerar_texto_embedding_receita(nome_receita, ingrediente, descricao, modo_preparo)
    embedding_vector = embed_text_api(texto_embedding)

    if embedding_vector:
        query.append({
            "$vectorSearch": {
                "queryVector": embedding_vector,
                "path": "embedding",
                "numCandidates": 100,
                "limit": 3,
                "index": "receita_embedding_index"
            }
        })

    if empresa_id:
        query.append({"$match": {"empresaId": empresa_id}})
    else:
        return {"status": "error", "data": "", "count": 0, "message": "ID da empresa não informado"}

    # (Opcional) projetar campos desejados
    query.append({
        "$project": {
            "_id": 0,
            "nome": 1,
            "ingredientes": 1,
            "modoPreparo": 1,
            "descricao": 1,
            "score": {"$meta": "vectorSearchScore"}  # retorna a similaridade
        }
    })

    receitas = []

    for doc in coll.aggregate(query):
        receitas.append({
            "nome": doc.get("nome"),
            "descricao": doc.get("descricao"),
            "ingredientes": {', '.join([i.get("nome") for i in doc.get("ingredientes", []) if i.get("nome")])},
            "modo_preparo": {' '.join(p.get('passo', '') for p in doc.get('modoPreparo', []))}
        })

    return {"status": "success", "data": receitas, "count": len(receitas)}

RECEITAS_TOOLS = [query_receitas]
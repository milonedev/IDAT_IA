import os
import json
import jwt
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

#################################### DEFINICION DE MODELOS

class LoginRequest(BaseModel):
    usuario: str
    contrasena: str

class PreguntaRequest(BaseModel): 
    question: str
    
class RespuestaResponse(BaseModel): 
    answer: str
    data_set: object
    
########################################################################
#Configurar las api-keys
os.environ["OPEN_API_KEY"] = "api_key_here"
SECRET_KEY = "12idat#llaveAutentication08032025"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_template = PromptTemplate(
    input_variables= ["chat_history", "question", "context", "user"],
    template="""
Eres EduMentor Seguro, un tutor virtual para universitarios creado por xAI. Responde basándote en el contexto proporcionado y los datos del usuario autenticado (id, nombre, preferencias, ubicación, cursos y notas). Sigue estas reglas:
- Responde de manera clara y concisa.
- Si no sabes la respuesta, di 'No tengo información suficiente'.
- Si la pregunta involucra datos no autorizados, di 'No tengo permiso para acceder a esa información por razones éticas'.
- Si un curso está "En curso" con nota < 60, añade: "¡Cuidado! Estás en riesgo de desaprobar este curso."
- El historial de conversación está disponible y tiene la estructura pregunta-respuesta.

Fecha actual: 08 de marzo de 2025.

Chat History:
{chat_history}

Usuario:
{user}

Contexto:
{context}

Pregunta:
{question}
"""

)
HISTORIAL = {}
AGENDAS = [
    {}
]
USUARIOS = {
    "juan123": {
        "user_id": 1,
        "user_name": "Juan Perez",
        "contrasena": "pass123",
        "preferencias": ["programación", "diseño web"],
        "location": "Lima, Perú",
        "cursos": {
            "Fisica 1" : {
                "nota": 80,
                "estado": "Aprobado",
                "creditos": 3,
                "profesor": "Pedro Marmol"
            },
            "Python Básico" : {
                "nota": 90,
                "estado": "Aprobado",
                "creditos": 3,
                "profesor": "Carlos Gómez"
            },
            "JavaScript Avanzado": {
                "nota": 10,
                "estado": "Desaprobado",
                "creditos": 4,
                "profesor": "Ana Torres"
            }
        }
    },
    "maria456": {
        "user_id": 2,
        "user_name": "María López",
        "contrasena": "pass456",
        "preferencias": ["ciencia de datos", "machine learning"],
        "location": "Madrid, España",
        "cursos": {
                "Análisis de Datos con Pandas": {
                "nota": 70,
                "estado": "Aprobado",
                "creditos": 3,
                "profesor": "Roberto Sánchez"
            },
            "Redes Neuronales" : {
                "nota": 65,
                "estado": "En curso",
                "creditos": 5,
                "profesor": "Elena Fernández"
            }
        }
    }
}
# Configurar LLM de OpenAI
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    api_key = os.environ["OPEN_API_KEY"],
    temperature = 0.7,
    max_tokens = 512,
)
# Configurar el metodo de autenticacion
security = HTTPBearer()

# funcion para generar el token
def create_token(user:str) -> str:
    payload = {"sub": user, "exp": datetime.utcnow() + timedelta(hours=1)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verificar_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        print('payload["sub"]', payload["sub"])
        return payload["sub"]
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="El token ha expirado.")
    except jwt.InvalidTokenError:
         raise HTTPException(status_code=401, detail="El token es invalido.")

# funcion para carfar y dividir el archivo de conociminetos
def load_and_preprocess_document(life_path): 
    loader = TextLoader(life_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size= 1000, chunk_overlap=200) # dividieno en fragmentos de 1000 caracteres
    texts = text_splitter.split_documents(documents=document)
    return texts

#Funcion para guardar el historial de conversacion 
def save_history(user_id, question, answer):
    if user_id not in HISTORIAL:
        HISTORIAL[user_id] = []
    
    HISTORIAL[user_id].append({"question": question, "answer": answer})
    
# Funcion para obtener el historial de conversacion
def get_history(user_id: int) -> str:
    if user_id not in HISTORIAL or not HISTORIAL[user_id]:
        return "No hay historial de conversacion."
    return "\n".join([f"Pregunta: {entry['question']}\n Respuesta: {entry['answer']}" for entry in HISTORIAL[user_id]])

#cofiguraicond e vector store
def create_vector_store(texts):
    if not texts:
        return ValueError('No existe texto a transformar')
    # los embedding se usan para la bsuqueda por similitud 
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="./chroma_dv2"
    )
    return vector_store

# ciclo de vida para manejar el core de la aplicacion
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store

    # cargar la base de conocimiento que es el .txt
    documents = load_and_preprocess_document('knowedge_base.txt')
    vector_store = create_vector_store(documents)
    print("Base de datos vectorial inicia")
    yield

app = FastAPI(
    title="IATutor safe API",
    description="Tutor virtual para universitarios",
    lifespan=lifespan
    )

# Fingir logeo con un uusario
@app.post('/login/')
async def login(request: LoginRequest):
    if not request.usuario:
        return {"message": "Por favor ingresar un usuario y contrasena"}
    
    user = request.usuario
    if user in USUARIOS and USUARIOS[user]["contrasena"] == request.contrasena:
        # generar el token
        token = create_token(user)
        
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Credenciales incorrectas.")

@app.post('/preguntar/', response_model=RespuestaResponse)
async def preguntar(request: PreguntaRequest, usuario: str = Depends(verificar_token)):
    try:
        user_data = USUARIOS[usuario]
        user_id = user_data["user_id"]
        user_name = user_data["user_name"]
        cursos = user_data["cursos"]
        DATA_COUNT = len(cursos)
        # print("DATA_COUNT", DATA_COUNT)
        cursos_info = "\n".join([f"{cursos}: Nota: {info['nota']}, Estado: {info['estado']}, Creditos: {info['creditos']}, profesor: {info['profesor']}" for curso, info in cursos.items() ])
        
        # El agente debe revisar si tengo una nota de desaprobacopm que es nota < 60 y  mandar una alerta
        advertencias = []
        
        # Buscar inforacion relevante en el vector store
        result = vector_store.similarity_search(request.question, k=3)
        context = "\n".join([doc.page_content for doc in result])
        
        # print('cursos.keys',cursos.keys())
        labels = list(cursos.keys())
        # print("labels", labels)
        notas = [curso['nota'] for curso in cursos.values()]
        # print("notas",notas)
        
        data_set = {
            "labels": labels[:DATA_COUNT],
            "datasets": [
                {
                "label": "Notas",
                "data": notas[:DATA_COUNT],
                "borderColor": "Utils.CHART_COLORS.blue",
                "backgroundColor": "Utils.transparentize(Utils.CHART_COLORS.blue, 0.5)",
                }
            ]
        }
        
        chat_history = get_history(user_id)

        # print("data_set", data_set)
        # Pasar en el template del prompt el contexto 
        full_query = prompt_template.format(context=context, question=request.question, user=cursos_info, chat_history=chat_history)
        
        # print("full query", full_query)
        response = llm.invoke(full_query)
        
        answer = f"Hola, {user_name}, la respuesta a tu consulta es: {response.content}"
        
        save_history(user_id, request.question, answer)

        return {"answer": answer, "data_set": data_set}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# get history by userid
@app.get('/api/history/{user_id}')
async def get_user_history(user_id: int, usuario: str = Depends(verificar_token)):
    print('USUARIOS[usuario]["user_id"]', USUARIOS[usuario]["user_id"])
    print('user_id', user_id)
    if USUARIOS[usuario]["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver el historial")
    history = get_history(user_id)
    return {"history": history}

@app.get('/')
async def root():
    return {"message": "IATutor is online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
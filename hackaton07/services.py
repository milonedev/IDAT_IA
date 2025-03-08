import jwt
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from config import SECRET_KEY, PROMPT_TEMPLATE
from models import UserData

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.environ["OPEN_API_KEY"],
    temperature=0.7,
    max_tokens=512,
)

def create_token(user: str) -> str:
    payload = {"sub": user, "exp": datetime.utcnow() + timedelta(hours=1)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verificar_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise Exception("El token ha expirado")
    except jwt.InvalidTokenError:
        raise Exception("El token es inválido")

def load_and_preprocess_document(file_path):
    loader = TextLoader(file_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents=document)

def create_vector_store(texts):
    if not texts:
        raise ValueError('No existe texto a transformar')
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return Chroma.from_documents(texts, embeddings, persist_directory="./chroma_dv2")

def save_history(user_id, question, answer):
    if user_id not in UserData.HISTORIAL:
        UserData.HISTORIAL[user_id] = []
    UserData.HISTORIAL[user_id].append({"question": question, "answer": answer})

def get_history(user_id: int) -> str:
    if user_id not in UserData.HISTORIAL or not UserData.HISTORIAL[user_id]:
        return "No hay historial de conversación."
    return "\n".join([f"Pregunta: {entry['question']}\nRespuesta: {entry['answer']}" 
                     for entry in UserData.HISTORIAL[user_id]])

async def process_question(question: str, user_data: dict, vector_store):
    user_id = user_data["user_id"]
    user_name = user_data["user_name"]
    cursos = user_data["cursos"]
    DATA_COUNT = len(cursos)
    
    cursos_info = "\n".join([f"{curso}: Nota: {info['nota']}, Estado: {info['estado']}, Créditos: {info['creditos']}, Profesor: {info['profesor']}" 
                           for curso, info in cursos.items()])
    
    result = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in result])
    
    labels = list(cursos.keys())
    notas = [curso['nota'] for curso in cursos.values()]
    
    data_set = {
        "labels": labels[:DATA_COUNT],
        "datasets": [{
            "label": "Notas",
            "data": notas[:DATA_COUNT],
            "borderColor": "Utils.CHART_COLORS.blue",
            "backgroundColor": "Utils.transparentize(Utils.CHART_COLORS.blue, 0.5)",
        }]
    }
    
    chat_history = get_history(user_id)
    full_query = PROMPT_TEMPLATE.format(context=context, question=question, user=cursos_info, chat_history=chat_history)
    response = llm.invoke(full_query)
    
    answer = f"Hola, {user_name}, la respuesta a tu consulta es: {response.content}"
    save_history(user_id, question, answer)
    
    return {"answer": answer, "data_set": data_set}
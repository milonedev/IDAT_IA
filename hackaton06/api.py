from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
import sqlite3

# Cargar api_key de openai
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
open_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

#conectar con la base de datos SQLite
conn = sqlite3.connect('users.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users(
    user_id INTEGER PRIMARY KEY,
    user_name TEXT NOT NULL,
    preferences TEXT NOT NULL,
    location TEXT NOT NULL
)
''')
conn.commit()

# history = []
history = {}

# 1.- Cargar y preparar el documento de kwowedge_base.txt
def load_and_preprocess_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_spliter.split_documents(documents)
    # print('Documentos cargados y procesados:')
    # for i, text in enumerate(texts):
    #     print(f"Documento ,{i+1}: {text.page_content[:100]}...")
    return texts

# 2.- Crear embeddings y almacenarlos en chromadb
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # agregar la validacion si no existe un texto a transformar 
    if not texts:
        raise ValueError('No hay texto para transformar.')

    sample_text = texts[0].page_content
    sample_embedding = embeddings.embed_query(sample_text)
    # print(f"Embedding de ejemplo: {sample_embedding}")

    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory="./chroma_dbv2"
    )

    return vector_store
prompt_template = PromptTemplate(
    input_variables=["context", "question", "user"],  # Variables que se usarán en el template
    template="""
Eres un asistente inteligente que responde preguntas basadas en el contexto proporcionado, adicional tendras acceso a informacion personal de un usuario, las respuestas deben ser claras y concisas referentes al contexto y al usuario, si no sabes la respuesta, di 'No tengo información suficiente'.
los datos que tienes del usuarios son el id, name, preferences y location. el usuario puede hacer consultas sobre su historial de conversaciones que se le proporcione el historial de conversacion tiene la estrucutra de pregunta y respuesta, donde pregunta es la consulta del usuario y la respuesta es el mensaje generado por la IA.

chat_history:
{chat_history}

user:
{user}

Contexto:
{context}

Pregunta:
{question}
"""
)
# 3.- Configurar el modelo geneerado de OpenAi
llm = ChatOpenAI(
    model = "gpt-3.5-turbo",
    api_key = open_api_key,
    temperature = 0.7,
    max_tokens = 512,
)
knowledge_base_file = 'knowedge_base.txt'
texts = load_and_preprocess_document(knowledge_base_file)
vector_store = create_vector_store(texts)

@app.route('/api/user', methods=['POST'])
def create_or_update_user():
    data = request.json
    user_id = data['user_id']
    user_name = data['user_name']
    preferences = data['preferences']
    location = data['location']

    if not user_id or not user_name:
        return jsonify({'error': 'user_id and user_name are required'}), 400

    cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
    exist_user = cursor.fetchone()
    if exist_user:
        cursor.execute('''update users set user_name = ?, preferences = ?, location = ? where user_id = ?''', (user_name, preferences, location, user_id))
    else:
        cursor.execute('''insert into users(user_id, user_name, preferences, location) values(?, ?, ?, ?)''', (user_id, user_name, preferences, location))

    conn.commit()

    return jsonify({'message': 'User created/updated successfully'}), 200

@app.route('/api/ask', methods=['Post'])
def ask():
    data = request.json
    user_id = data['user_id']
    question = data['question']

    if not user_id or not question:
        return jsonify({'error': 'user_id and question are required'}), 400
    
    cursor.execute("SELECT user_id, user_name, preferences, location FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()

    if not user:
        return jsonify({'error': 'User not found'}), 404
    user_id, user_name, preferences, location = user

    result = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in result])
    # full_query = f"Contexto: {context}\n Pregunta: {question}"
    
    chat_history = get_history(user_id)

    full_query = prompt_template.format(context=context, question=question, user=user, chat_history=chat_history)
    response = llm.invoke(full_query)

    answer = f"Hola, {user_name}, la respuesta a tu pregunta es: {response.content}"

    if user_id not in history:
        history[user_id] = []
    history[user_id].append({'question': question, 'answer': answer})

    return jsonify({'answer': answer}), 200

@app.route('/api/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    if user_id not in history or not history[user_id]:
        return jsonify({'error': 'No hay un historial de conversacion'}), 404

    return jsonify({'history': history[user_id]}), 200

if(__name__ == '__main__'):
    app.run(debug=True)
import os

from deep_translator import GoogleTranslator
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Updated imports based on LangChain v0.2+
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify, send_file
from langchain.memory import ConversationBufferWindowMemory
from gtts import gTTS
# 1.- COnfiguramos del embedding
embedings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# # 2.- convertir las descripciones de los productos en vectores, embeddings y almacenarlos en un vector store
# doc = [Document(page_content= producto["descripcion"] , metadata={"nombre": producto["nombre"], "precio": producto["precio"], "cantidad": producto["cantidad"]}) for producto in productos]

# TRAER TODOS LOS PRODUCTOS DE LA API
def fetch_prodcuts():
    url = 'https://api.escuelajs.co/api/v1/products?offset=0&limit=5'
    response = requests.get(url)

    print(response)
    if (response.status_code == 200):
        products = response.json()

        doc = [Document(
            page_content=(
                f"title: {producto['title']}."
                f"price: {producto['price']}."
                f"description: {producto['description']}."
                f"category: {producto['category']['name']}."
            ), 
            metadata={
                "title": producto["title"], 
                "price": producto["price"]
            }) for producto in products]

        return doc
    else:
        return Exception("Error al obtener los productos")

docs = fetch_prodcuts()
# 3.- crear y guardar el vector dentro de una bd 
vectorstore = Chroma.from_documents(docs, embedings)

# 4.- configuracion de la memoria para mantener la conversacion
# (Limitar la memoria a los ultimos 3 mensajes)
memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    input_key= "question",
    output_key= "answer"
)

# (no liimita el historial de memoria de los mensjes)
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True,
#     input_key= "question",
#     output_key= "answer"
# )

prompt_template = """=
Identify the language and answer the query in that same language.
You are a chatbot that helps users find products in an online store.
if you don't know or the data provided is not enough to answer a specific question, just answer “I don't have an answer for that question”.
the products found in the store are: {context}

chat history: {chat_history}
question: {question}
Useful answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)
# 5.- crea ek sistema de recuperacion conversacional con RAG
llm = OpenAI(temperature=0,openai_api_key='')
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose = True,
    output_key = "answer",
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

def chat_with_bot(query):
    print('qa:', qa)
    result = qa.invoke({"question": query})

    print("result:", result.get("source_documents", []))

    if not result.get("source_documents"):
        return "No tengo respuesta para esa pregunta"
    
    # interceptar la respuesta y transformarla al idioma español
    print('result["answer"]', result["answer"])
    # translate_response = GoogleTranslator(source='en', target='es').translate(result["answer"])
    return result["answer"]

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('question')
        if not query:
            return jsonify({
                'error': 'No question provided'
            }), 400

        # interceptar la pregunta y transformarla al idioma ingles
        # transtlate_query = GoogleTranslator(source='es', target='en').translate(query)
        # print('transtlate_query', transtlate_query)

        # response = chat_with_bot(query)
        print('tts')
        tts = gTTS(text=query, lang='es')
        print(tts)
        print('app.config["UPLOAD_FOLDER"]',app.config["UPLOAD_FOLDER"])
        audio_response = os.path.join(app.config["UPLOAD_FOLDER"], 'response.mp3')
        print(audio_response)
        tts.save(audio_response)

        print('audio_response:', audio_response)

        # return jsonify({
        #     'response': query
        # }), 200
        return send_file(audio_response, mimetype='audio/mp3')
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True,port=5000)
from langchain_openai import ChatOpenAI 
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Cargar api_key de openai
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
open_api_key = os.getenv("OPENAI_API_KEY")

# 1.- Cargar y preparar el documento
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
        persist_directory="./chroma_db"
    )

    return vector_store
# 3.- Configurar el modelo geneerado de OpenAi
def setup_generative_model():
    llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        api_key = open_api_key,
        temperature = 0.7,
        max_tokens = 512,
    )

    return llm
# 4.- Crear la función de ordenamiento con un re ranking
def rerank_documents(query, documents, top_k=3):
    # print('query', query)
    # print('documents', documents)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc.page_content) for doc in documents]
    # print('pairs', pairs)
    scores  = cross_encoder.predict(pairs)
    # print('Documents: ', documents)
    score_docs = list(zip(documents, scores))
    score_docs.sort(key=lambda x: x[1], reverse=True)
    top_documents = [doc for doc, score in score_docs[:top_k]]

    # print('Top documents: ', top_documents)
    return top_documents
# 5.- Crear el agente(RAG)
def create_rag_agent(vector_store, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vector_store.as_retriever(
            search_kwargs = {"k": 10}
        ),
        return_source_documents = True
    )

    return qa_chain
# 6.- Funcion para consiltar el agente con re ranking
def ask_question(qa_chain, question):
    result = qa_chain.invoke({"query": question})
    source_documents = result["source_documents"]

    # Hacer el re ranked para quedarnos con la informacion mas preciza
    reranked_documents = rerank_documents(question, source_documents, top_k=3)

    # print(f"reranked_documents: {reranked_documents}")
    print(f"Respuesta: {result['result']}")
    for doc in reranked_documents:
        print(f"Documento: {doc.page_content[:400]}...")

# 7.- Ejecturar el flujo
if __name__ == '__main__':
    file_path = 'knowedge_base.txt'
    texts = load_and_preprocess_document(file_path)
    vector_store = create_vector_store(texts)
    llm = setup_generative_model()
    qa_chain = create_rag_agent(vector_store, llm)
    question = "¿Cual es la tasa de interes para un credito automotriz?"
    ask_question(qa_chain, question)


# Tasas de interés para créditos:
# - Crédito personal: 15% anual.
# - Crédito hipotecario: 8% anual.
# - Crédito automotriz: 10% anual.

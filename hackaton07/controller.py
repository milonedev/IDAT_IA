from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
from models import LoginRequest, PreguntaRequest, RespuestaResponse, UserData
from services import create_token, verificar_token, load_and_preprocess_document, create_vector_store, process_question, get_history

security = HTTPBearer()
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    documents = load_and_preprocess_document('knowedge_base.txt')
    vector_store = create_vector_store(documents)
    print("Base de datos vectorial iniciada")
    yield

app = FastAPI(
    title="IATutor Safe API",
    description="Tutor virtual para universitarios",
    lifespan=lifespan
)

@app.post('/login/')
async def login(request: LoginRequest):
    if not request.usuario:
        return {"message": "Por favor ingresar un usuario y contraseña"}
    
    user = request.usuario
    if user in UserData.USUARIOS and UserData.USUARIOS[user]["contrasena"] == request.contrasena:
        token = create_token(user)
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Credenciales incorrectas")

@app.post('/preguntar/', response_model=RespuestaResponse)
async def preguntar(request: PreguntaRequest, credentials=Depends(security)):
    try:
        usuario = verificar_token(credentials.credentials)
        user_data = UserData.USUARIOS[usuario]
        return await process_question(request.question, user_data, vector_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get('/api/history/{user_id}')
async def get_user_history(user_id: int, credentials=Depends(security)):
    usuario = verificar_token(credentials.credentials)
    if UserData.USUARIOS[usuario]["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver el historial")
    history = get_history(user_id)
    return {"history": history}

@app.get('/')
async def root():
    return {"message": "IATutor is online"}
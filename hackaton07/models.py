from pydantic import BaseModel
from typing import Optional, Dict

class LoginRequest(BaseModel):
    usuario: str
    contrasena: str

class PreguntaRequest(BaseModel): 
    question: str
    
class RespuestaResponse(BaseModel): 
    answer: str
    data_set: object

class UserData:
    USUARIOS = {
        "juan123": {
            "user_id": 1,
            "user_name": "Juan Perez",
            "contrasena": "pass123",
            "preferencias": ["programación", "diseño web"],
            "location": "Lima, Perú",
            "cursos": {
                "Fisica 1": {"nota": 80, "estado": "Aprobado", "creditos": 3, "profesor": "Pedro Marmol"},
                "Python Básico": {"nota": 90, "estado": "Aprobado", "creditos": 3, "profesor": "Carlos Gómez"},
                "JavaScript Avanzado": {"nota": 10, "estado": "Desaprobado", "creditos": 4, "profesor": "Ana Torres"}
            }
        },
        "maria456": {
            "user_id": 2,
            "user_name": "María López",
            "contrasena": "pass456",
            "preferencias": ["ciencia de datos", "machine learning"],
            "location": "Madrid, España",
            "cursos": {
                "Análisis de Datos con Pandas": {"nota": 70, "estado": "Aprobado", "creditos": 3, "profesor": "Roberto Sánchez"},
                "Redes Neuronales": {"nota": 65, "estado": "En curso", "creditos": 5, "profesor": "Elena Fernández"}
            }
        }
    }
    
    HISTORIAL = {}
    AGENDAS = [{}]
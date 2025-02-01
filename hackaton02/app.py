from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import fitz # extraer texto del pdf
from pdf2image import convert_from_path
from flask_sqlalchemy import SQLAlchemy
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'static/images'

if not os.path.exists(UPLOAD_FOLDER) or not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pdfs.db'
app.secret_key = 'secreto'
 
db = SQLAlchemy(app)

class PDFFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable = False)
    content = db.Column(db.Text, nullable = False)
# Servir una aplicaion para el frontend
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file: 
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        images = convert_from_path(filepath)
        if images:
            image_path = os.path.join(app.config['IMAGE_FOLDER'], f"{file.filename}.png")
            images[0].save(image_path, "PNG")
        # metodo para extraer texto del pdf
        doc = fitz.open(filepath)
        content = "\n".join([page.get_text() for page in doc])

        # guardar en la base de datos.
        pdf_entry = PDFFile(filename=file.filename, content = content)
        db.session.add(pdf_entry)
        db.session.commit()

        # guardar una sesion 
        session['pdf_id'] = pdf_entry.id

        return redirect(url_for('chat', filename = file.filename))
    return redirect(request.url)

@app.route('/chat')
def chat():
    pdf_id = session.get('pdf_id')
    print('pdf_id', pdf_id)
    if pdf_id:
        pdf = PDFFile.query.get(pdf_id)
        print('pdf', pdf)
        return render_template('chat.html', pdf_content= pdf.content, filename = pdf.filename)
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get('question')
    pdf_id = session.get('pdf_id')

    if not pdf_id:
        return jsonify({"error": "No hay pdf asociado a este id"}), 400
    
    pdf = PDFFile.query.get(pdf_id)

    openai_url = "https://api.openai.com/v1/chat/completions"
    # openai_url = "https://api.openai.com/v1/completions"
          
    headers = {
        "Authorization": "Bearer myToken",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            { "role": "system", "content": f"Analiza el texto y responde las preguntas basándote en el siguiente texto: {repr(pdf.content)}" },
            { "role": "user", "content": user_question }
        ],
        "temperature": 0.7
    }

    response  = requests.post(openai_url, json=payload, headers = headers)

    print('response: ',response.json())

    response_data = response.json()
    return jsonify({"response": response_data.get("choices", [{}])[0].get("message", {}).get("content", "error en la respuesta")})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
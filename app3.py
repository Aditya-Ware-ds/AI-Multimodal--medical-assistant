from flask import Flask, render_template, request, jsonify
import os
import pickle
import faiss
import numpy as np
import speech_recognition as sr  # Import SpeechRecognition
from sentence_transformers import SentenceTransformer
from ollama import Client
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "medllama2"
INDEX_FILE = "first_aid_index.faiss"
CHUNKS_FILE = "first_aid_chunks.pkl"

# RAG Assistant
class FirstAidAssistant:
    def __init__(self):
        # Load embeddings
        self.index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, 'rb') as f:
            self.chunks = pickle.load(f)
        # Initialize models
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.ollama = Client(host='http://localhost:11434')

    def search(self, query, top_k=3):
        # Get relevant chunks
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        return [self.chunks[i] for i in indices[0]]

    def ask(self, wound_type):
        # Retrieve context
        context_chunks = self.search(wound_type)
        context = "\n".join(context_chunks)
        # Generate answer
        response = self.ollama.generate(
            model=LLM_MODEL,
            prompt=f"Use this first aid information to answer: {context}\n\nQuestion: How to treat a {wound_type}?",
            stream=False
        )
        return response['response']

# Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route('/api/ask', methods=['POST'])
def handle_question():
    try:
        data = request.get_json()
        user_question = data.get('question', '')

        print(f"\nReceived text input: {user_question}")
        assistant = FirstAidAssistant()
        answer = assistant.ask(user_question)

        return jsonify({'response': answer})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------- IMAGE CLASSIFICATION + FIRST AID --------------------------------

# Load the trained model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)  # 7 classes
model.load_state_dict(torch.load(r"C:\Users\adity\Desktop\Google solution 2025\AI-multimodal-Medical-assistant-main\wound_image_classification_model.pth"))
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names (Ensure they match your dataset)
class_names = ['Abrasions', 'Bruises', 'Burns', 'Cut', 'Ingrown_nails', 'Laceration', 'Stab_wound']

def predict_image(image, model):
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations & move to GPU

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get highest probability class
        class_naam = class_names[predicted.item()]
        print(class_naam)
    
    return class_naam

@app.route('/api/analyze-image', methods=['POST'])
def handle_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")  # Open image in RGB mode

        # Get wound type prediction
        predicted_class = predict_image(image, model)

        # Use the wound type to query the RAG + LLM assistant
        assistant = FirstAidAssistant()
        prompt = f"I got a {predicted_class}, help me to first aid it."
        answer = assistant.ask(prompt)
        return jsonify({'response': answer})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --------------------- VOICE INPUT USING SpeechRecognition ------------------------------------

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # Record the entire audio file

    try:
        text = recognizer.recognize_google(audio)  # Use Google Web Speech API
        print(f"Transcribed Text: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Error with the speech recognition service."

@app.route('/api/voice', methods=['POST'])
def handle_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400

        audio_file = request.files['audio']
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)

        # Convert speech to text
        text_query = speech_to_text(audio_path)
        print(text_query)

        # Get first aid response from RAG + LLM
        assistant = FirstAidAssistant()
        first_aid_info = assistant.ask(text_query)

        return jsonify({'response': first_aid_info})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

import requests
import json
from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Neo4j database connection details
URI = "neo4j://localhost"
AUTH = ("neo4j", "Password")

# Ollama API endpoint (adjust URL if necessary)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
headers = {
    "Content-Type": "application/json"
}

# Function to send a request to the Ollama API
def send_to_ollama(prompt):
    payload = {
        "model": "supachai/llama-3-typhoon-v1.5",  # Updated model name
        "prompt": prompt + "ตอบแบบทันตแพทย์ชาย และไม่เกิน 20 คำ",
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        data = json.loads(response.text)
        return f"{data['response']}"
    else:
        return f"Failed to get a response from Ollama: {response.status_code}, {response.text}"

# Function to run Neo4j queries
def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

# Load greeting and question corpus from Neo4j
def load_corpora():
    greeting_query = '''
    MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
    '''
    question_query = '''
    MATCH (q:Question) RETURN q.name as name, q.msg_reply as reply;
    '''
    
    greetings = run_query(greeting_query)
    questions = run_query(question_query)
    
    greeting_corpus = [record['name'] for record in greetings]
    question_corpus = [record['name'] for record in questions]
    
    greeting_dict = {record['name']: record['reply'] for record in greetings}
    question_dict = {record['name']: record['reply'] for record in questions}
    
    return greeting_corpus, question_corpus, greeting_dict, question_dict

greeting_corpus, question_corpus, greeting_dict, question_dict = load_corpora()

# Function to compute cosine similarity between input sentence and corpus
def compute_response(sentence):
    # Encode the question corpus and the input sentence
    question_vec = model.encode(question_corpus, convert_to_tensor=True, normalize_embeddings=True)
    ask_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    
    # Compute cosine similarity
    question_scores = util.cos_sim(question_vec, ask_vec).cpu().numpy()
    
    # Get the index of the most similar question
    max_question_index = np.argmax(question_scores)
    max_question_score = question_scores[max_question_index]
    
    # Check if the similarity score is greater than a threshold (0.5)
    if max_question_score > 0.5:
        Match_question = question_corpus[max_question_index]
        return f"จาก Neo4J เกี่ยวกับเรื่องฟันผุ\nคำถาม: {sentence}\nคำตอบ: {question_dict.get(Match_question, 'ไม่พบคำตอบในระบบ')}"
    else:
        # Handle greeting
        greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
        greeting_scores = util.cos_sim(greeting_vec, ask_vec).cpu().numpy()
        
        max_greeting_index = np.argmax(greeting_scores)
        max_greeting_score = greeting_scores[max_greeting_index]
        
        if max_greeting_score > 0.5:
            Match_greeting = greeting_corpus[max_greeting_index]
            return f"จาก Neo4J คำถามทั่วไป\nคำถาม: {sentence}\nคำตอบ: {greeting_dict.get(Match_greeting, 'ไม่พบคำตอบในระบบ')}"
        else:
            # Send to Ollama API if no match found
            return f"จาก Ollama\nคำถาม: {sentence}\nคำตอบ: " + send_to_ollama(sentence)

# Flask app to handle Line bot messages
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'vc8ZqAx20Fd0+CXx0JaqH8dPP2plIAf6YYig4J1l8sGiDaiyicBIzG4cV+0ELu46OeeybvGp3jIUFhKzcCKnfOnNh2+eUnvggvUxMt8AL5yiC1OfYRCJJ1n90BmXnT+hrxDIV33lOIOQDJiuBU3TTQdB04t89/1O/w1cDnyilFU='
        secret = 'e4b3adab91258b1e658cf1cdee43e5a4'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        
        # Get the message and reply token from the JSON data
        msg = json_data['events'][0]['message']['text']
        tk = json_data['events'][0]['replyToken']
        
        # Compute the response based on the message
        response_msg = compute_response(msg)
        
        # Reply to the user via Line
        line_bot_api.reply_message(tk, TextSendMessage(text=response_msg))
        print(msg, tk)
    except Exception as e:
        print(f"Error: {e}")
        print(body)
    return 'OK'

if __name__ == '__main__':
    # For Debugging
    compute_response("ฟันผุคืออะไร?")
    app.run(port=8080)
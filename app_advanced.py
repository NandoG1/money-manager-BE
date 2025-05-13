from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get API key from environment variable
API_KEY = os.getenv('GEMINI_API_KEY')

# Gemini API endpoint
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Financial advisor context
context = """You are a financial advisor AI, helping users with personal finance management. Your role is to provide practical advice on budgeting, saving, investing, and managing debt. 
Always reply in the same language the user used (English or Indonesian). Keep answers brief and helpful, like a mini-chatbot. 
Adapt suggestions based on user input, such as income, expenses, or financial goals."""

# Financial planner context (for one-time advice)
financial_planner_context = """You are a financial planning AI assistant. Based on the user's income, expenses, and financial goals, provide concise and actionable financial advice. 
Focus on budget optimization, savings strategies, and goal achievement. Structure your response to be easy to understand with bullet points for actionable steps.
Always respond in the same language as the user (English or Indonesian). Limit your response to 3-5 key recommendations that are most impactful for their situation."""

# Store conversation history per session
conversations = {}

def get_response_from_gemini(user_input, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    
    # Build prompt with conversation history
    prompt = f"{context}\n"
    
    # Add conversation history
    for entry in conversation_history:
        if entry['role'] == 'user':
            prompt += f"User: {entry['content']}\n"
        else:
            prompt += f"AI: {entry['content']}\n"
    
    # Add current user input
    prompt += f"User: {user_input}\nAI:"
    
    # Data to send to API
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
    }

    # Send POST request to Gemini API
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    
    # Check status and get result
    if response.status_code == 200:
        result = response.json()
        output_text = result['candidates'][0]['content']['parts'][0]['text']
        return output_text
    else:
        return f"Error: {response.status_code} - {response.text}"

def get_financial_advice(income, expenses, goals):
    # Create a prompt for financial advice
    prompt = f"{financial_planner_context}\n\nUser Information:\nMonthly Income: {income}\nMonthly Expenses: {expenses}\nFinancial Goals: {goals}\n\nProvide financial advice:"
    
    # Data to send to API
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
    }

    # Send POST request to Gemini API
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    
    # Check status and get result
    if response.status_code == 200:
        result = response.json()
        output_text = result['candidates'][0]['content']['parts'][0]['text']
        return output_text
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('sessionId', str(uuid.uuid4()))
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get or create conversation history
    if session_id not in conversations:
        conversations[session_id] = []
    
    # Add user message to history
    conversations[session_id].append({
        'role': 'user',
        'content': user_input
    })
    
    # Get response using conversation history
    ai_response = get_response_from_gemini(user_input, conversations[session_id])
    
    # Add AI response to history (limit history to last 10 messages to avoid token limits)
    conversations[session_id].append({
        'role': 'assistant',
        'content': ai_response
    })
    
    # Keep only the last 10 messages in history
    if len(conversations[session_id]) > 10:
        conversations[session_id] = conversations[session_id][-10:]
    
    return jsonify({
        'response': ai_response,
        'sessionId': session_id
    })

@app.route('/api/financial-advice', methods=['POST'])
def financial_advice():
    data = request.json
    
    # Get financial information from request
    income = data.get('income', '')
    expenses = data.get('expenses', '')
    goals = data.get('goals', '')
    
    # Validate input
    if not income or not expenses:
        return jsonify({'error': 'Income and expenses are required'}), 400
    
    # Get one-time financial advice (no history)
    advice = get_financial_advice(income, expenses, goals)
    
    return jsonify({
        'advice': advice
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    session_id = request.args.get('sessionId', '')
    
    if not session_id or session_id not in conversations:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    return jsonify({
        'history': conversations[session_id],
        'sessionId': session_id
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
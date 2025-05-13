from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
import io
import base64
from dotenv import load_dotenv
import uuid
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
from datetime import datetime, timedelta

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

def round_to_multiple(value, multiple):
    return round(value / multiple) * multiple


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

def extract_text_from_image(image_data):
    """Extract text from image using OCR"""
    try:
        # Set OCR configuration
        custom_config = r'--oem 3 --psm 6'
        
        # Process the image with pytesseract
        text = pytesseract.image_to_string(image_data, lang='ind', config=custom_config)
        return text
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return None

def analyze_receipt(ocr_text):
    """Send OCR text to Gemini API for analysis"""
    prompt = f"""
Berikut adalah isi struk hasil OCR:

{ocr_text}

Tolong ekstrak informasi penting dari struk ini dan berikan hasil dalam format JSON dengan key:
- "judul": judul transaksi yang relevan (kalo ga ada bisa liat daftar belinya dan cari judul yang tepat)
- "tanggal": tanggal transaksi aja tidak usah ada jam waktunya(format bebas, tapi jelas)
- "subtotal": nilai total pembayaran
Jika tidak ditemukan, isi dengan "Tidak ditemukan".

PENTING: Berikan response dalam format JSON yang valid tanpa backtick, tanpa markdown, tanpa penjelasan, hanya object JSON saja. Example: {{"judul": "Supermarket ABC", "tanggal": "15/06/2023", "subtotal": "250000"}}
    """

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
        output_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Ensure we have valid JSON by extracting it if needed
        try:
            # Try to parse as JSON directly first
            json.loads(output_text)
            return output_text
        except json.JSONDecodeError:
            # If it's not valid JSON, try to extract JSON object
            import re
            json_pattern = r'{.*}'
            match = re.search(json_pattern, output_text, re.DOTALL)
            
            if match:
                extracted_json = match.group(0)
                try:
                    # Validate extracted JSON
                    json.loads(extracted_json)
                    return extracted_json
                except:
                    # If still not valid, return fallback JSON
                    return '{"judul": "Tidak ditemukan", "tanggal": "Tidak ditemukan", "subtotal": "Tidak ditemukan"}'
            else:
                # Fallback if no JSON object found
                return '{"judul": "Tidak ditemukan", "tanggal": "Tidak ditemukan", "subtotal": "Tidak ditemukan"}'
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

@app.route('/api/ocr-receipt', methods=['POST'])
def ocr_receipt():
    # Check if image is provided
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Get base64 encoded image from request
        image_data = request.json.get('image', '')
        
        # Decode base64 string to image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract text using OCR
        ocr_text = extract_text_from_image(image)
        
        if not ocr_text:
            return jsonify({'error': 'Failed to extract text from image'}), 500
        
        # Analyze receipt using AI
        analysis_result = analyze_receipt(ocr_text)
        
        # Return OCR text and analysis
        return jsonify({
            'ocr_text': ocr_text,
            'analysis': analysis_result
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
@app.route('/api/predict', methods=['POST'])
def predict_finances():
    # Get transaction data
    data = request.json
    
    # Set random seeds for consistent results
    random.seed(42)
    np.random.seed(42)
    
    if not data or not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Invalid data format. Expected non-empty array of transactions"}), 400

    try:
        # Convert transaction data to DataFrame
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = ['date', 'income', 'expense']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({"error": f"Missing required column: {col}"}), 400
        
        # Convert date to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort data by date
        df = df.sort_values('date')
        
        # Add 'days_since_start' column for prediction model
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        # Add additional features to improve prediction
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        
        # Add lag features (values from previous days)
        for lag in range(1, 4):  # Use data from previous 3 days
            if len(df) > lag:
                df[f'income_lag_{lag}'] = df['income'].shift(lag).fillna(df['income'].mean())
                df[f'expense_lag_{lag}'] = df['expense'].shift(lag).fillna(df['expense'].mean())

        # For rolling average calculation
        df['income_rolling_mean'] = df['income'].rolling(window=7, min_periods=1).mean()
        df['expense_rolling_mean'] = df['expense'].rolling(window=7, min_periods=1).mean()
        
        # Define features for the model
        features = ['days_since_start', 'day_of_week', 'month', 
                    'income_rolling_mean', 'expense_rolling_mean']
        
        # Add lag features if available
        for lag in range(1, 4):
            if f'income_lag_{lag}' in df.columns:
                features.append(f'income_lag_{lag}')
                features.append(f'expense_lag_{lag}')
        
        # Initialize the models with fixed random state
        income_model = RandomForestRegressor(n_estimators=100, random_state=42)
        expense_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Prepare data for models
        X = df[features]
        
        # Train income model
        income_model.fit(X, df['income'])
        
        # Train expense model
        expense_model.fit(X, df['expense'])
        
        # Predict for the next 7 days
        future_dates = [df['date'].max() + timedelta(days=i) for i in range(1, 8)]
        
        # Prepare future features
        future_features = []
        
        last_income_values = list(df['income'].tail(3))
        last_expense_values = list(df['expense'].tail(3))
        last_income_rolling = df['income_rolling_mean'].iloc[-1]
        last_expense_rolling = df['expense_rolling_mean'].iloc[-1]
        last_day = df['days_since_start'].max()
        
        # Define rounding step (5000 for rounding to the nearest 5000)
        rounding_step = 5000
        
        for i, future_date in enumerate(future_dates):
            day_future = {
                'days_since_start': last_day + i + 1,
                'day_of_week': future_date.dayofweek,
                'month': future_date.month,
                'income_rolling_mean': last_income_rolling,
                'expense_rolling_mean': last_expense_rolling
            }
            
            # Add lag features
            for lag in range(1, 4):
                if i >= lag:
                    # Use previous predictions
                    day_future[f'income_lag_{lag}'] = future_features[i-lag]['predicted_income']
                    day_future[f'expense_lag_{lag}'] = future_features[i-lag]['predicted_expense']
                else:
                    # Use historical data
                    day_future[f'income_lag_{lag}'] = last_income_values[-lag]
                    day_future[f'expense_lag_{lag}'] = last_expense_values[-lag]
            
            # Make predictions for this day
            future_X = pd.DataFrame([{k: v for k, v in day_future.items() if k in features}])
            
            predicted_income = income_model.predict(future_X)[0]
            predicted_expense = expense_model.predict(future_X)[0]
            
            # Add controlled random variation to make predictions more realistic
            # Use a fixed random seed to ensure consistency
            # You can comment these 4 lines out if you want exactly the same predictions every time without randomness
            income_std = df['income'].std() * 0.15  # Use 15% of the standard deviation
            expense_std = df['expense'].std() * 0.15
            predicted_income += random.normalvariate(0, income_std)
            predicted_expense += random.normalvariate(0, expense_std)
            
            # Ensure predictions are not negative
            predicted_income = max(0, predicted_income)
            predicted_expense = max(0, predicted_expense)
            
            # Round to the nearest multiple of rounding_step
            predicted_income = round_to_multiple(predicted_income, rounding_step)
            predicted_expense = round_to_multiple(predicted_expense, rounding_step)
            
            # Update rolling averages for next predictions
            last_income_rolling = (last_income_rolling * 6 + predicted_income) / 7
            last_expense_rolling = (last_expense_rolling * 6 + predicted_expense) / 7
            
            # Store this prediction
            day_result = {
                "date": future_date.strftime('%Y-%m-%d'),
                "predicted_income": int(predicted_income),  # Convert to integer
                "predicted_expense": int(predicted_expense)  # Convert to integer
            }
            
            future_features.append(day_result)
        
        return jsonify({"prediction": future_features})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
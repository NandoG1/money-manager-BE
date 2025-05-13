# Financial Advisor Chatbot Backend

A Flask-based backend API for a financial advisor chatbot that uses Google's Gemini API.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. Install Tesseract OCR:
   - For Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - For macOS: `brew install tesseract`
   - For Windows: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install Indonesian language pack as needed
5. Run the server:
   ```
   python run.py
   ```

## API Endpoints

### Chat Endpoint (with history)

`POST /api/chat`

Request body:

```json
{
	"message": "How can I save more money?"
}
```

Response:

```json
{
	"response": "To save more money, you can start by...",
	"sessionId": "12345-67890-abcde"
}
```

### One-Time Financial Advice Endpoint

`POST /api/financial-advice`

Request body:

```json
{
	"income": "5000000",
	"expenses": "3500000",
	"goals": "Buying a house in 5 years and saving for retirement"
}
```

Response:

```json
{
	"advice": "Based on your financial situation, here are my recommendations: ..."
}
```

### OCR Receipt Processing

`POST /api/ocr-receipt`

Request body:

```json
{
	"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
}
```

Response:

```json
{
	"ocr_text": "Raw OCR text extracted from the receipt...",
	"analysis": "{\"judul\":\"Supermarket ABC\",\"tanggal\":\"15/06/2023\",\"subtotal\":\"250000\"}"
}
```

### History Endpoint

`GET /api/history?sessionId=12345-67890-abcde`

Response:

```json
{
	"history": [
		{ "role": "user", "content": "How can I save money?" },
		{ "role": "assistant", "content": "To save money, you can..." }
	],
	"sessionId": "12345-67890-abcde"
}
```

### Health Check

`GET /api/health`

Response:

```json
{
	"status": "ok"
}
```

## Frontend Integration (Next.js)

Example of how to connect from a Next.js frontend:

```typescript
// Example API call from Next.js
const sendMessage = async (message: string) => {
	try {
		const response = await fetch("http://localhost:5000/api/chat", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ message }),
		});

		const data = await response.json();
		return data.response;
	} catch (error) {
		console.error("Error sending message:", error);
		return "Sorry, there was an error processing your request.";
	}
};
```

### Example for Financial Advice Endpoint

```typescript
const getFinancialAdvice = async (income: string, expenses: string, goals: string) => {
	try {
		const response = await fetch("http://localhost:5000/api/financial-advice", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ income, expenses, goals }),
		});

		const data = await response.json();
		return data.advice;
	} catch (error) {
		console.error("Error getting financial advice:", error);
		return "Sorry, there was an error processing your request.";
	}
};
```

### Example for OCR Receipt Processing

```typescript
const processReceipt = async (imageBase64: string) => {
	try {
		const response = await fetch("http://localhost:5000/api/ocr-receipt", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ image: imageBase64 }),
		});

		const data = await response.json();
		// Parse the JSON string from the analysis field
		const analysis = JSON.parse(data.analysis);
		return {
			rawText: data.ocr_text,
			judul: analysis.judul,
			tanggal: analysis.tanggal,
			subtotal: analysis.subtotal,
		};
	} catch (error) {
		console.error("Error processing receipt:", error);
		return null;
	}
};
```

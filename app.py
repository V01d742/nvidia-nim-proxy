from flask import Flask, request, jsonify, Response
import requests
import json
import os

app = Flask(__name__)

# Configuration
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', 'your-nvidia-api-key-here')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
        # Map OpenAI request to NVIDIA NIM format
        nvidia_request = {
            "model": data.get("model", "meta/llama-3.1-8b-instruct"),
            "messages": data.get("messages", []),
            "temperature": data.get("temperature", 0.7),
            "top_p": data.get("top_p", 1.0),
            "max_tokens": data.get("max_tokens", 1024),
            "stream": data.get("stream", False)
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Handle streaming
        if nvidia_request["stream"]:
            return handle_streaming(nvidia_request, headers)
        
        # Non-streaming request
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json=nvidia_request,
            timeout=60
        )
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_streaming(nvidia_request, headers):
    """Handle streaming responses"""
    def generate():
        try:
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=nvidia_request,
                stream=True,
                timeout=60
            )
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        yield decoded_line + '\n\n'
                        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{NVIDIA_BASE_URL}/models",
            headers=headers,
            timeout=30
        )
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        # Fallback model list
        return jsonify({
            "object": "list",
            "data": [
                {"id": "meta/llama-3.1-8b-instruct", "object": "model"},
                {"id": "meta/llama-3.1-70b-instruct", "object": "model"},
                {"id": "mistralai/mixtral-8x7b-instruct-v0.1", "object": "model"}
            ]
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "OpenAI to NVIDIA NIM Proxy",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/models",
            "/health"
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

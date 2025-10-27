from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_adapter import GemmaModelLoader
from chat_handler import TherapeuticChatHandler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables for model and chat handler
model_loader = None
chat_handler = None

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FLASK_APP_PATH = os.environ.get("FLASK_APP_PATH", BASE_DIR)
TEMPLATE_PATH = os.environ.get("FLASK_TEMPLATE_PATH", os.path.join(FLASK_APP_PATH, "templates"))
STATIC_PATH = os.environ.get("FLASK_STATIC_PATH", os.path.join(FLASK_APP_PATH, "static"))

# Update Flask template and static folders
app.template_folder = TEMPLATE_PATH
app.static_folder = STATIC_PATH

def initialize_model():
    """Initialize the Gemma 3n therapeutic model on startup"""
    global model_loader, chat_handler
    
    try:
        print("\n" + "=" * 70)
        print("Starting Therapeutic Chatbot Server")
        print("=" * 70)
        
        # Load model with correct paths
        model_loader = GemmaModelLoader()
        
        if model_loader.initialize():
            chat_handler = TherapeuticChatHandler(model_loader)
            print("\n" + "=" * 70)
            print("✓ Flask app ready to serve therapeutic conversations!")
            print(f"✓ Server running at: http://localhost:5000")
            print("=" * 70 + "\n")
            return True
        else:
            print("\n✗ Model initialization failed")
            return False
            
    except Exception as e:
        print(f"\n✗ Initialization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Serve the main therapeutic chat interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with model information"""
    if model_loader and chat_handler:
        return jsonify({
            "status": "healthy",
            "model_info": model_loader.get_model_info(),
            "conversation_stats": chat_handler.get_conversation_summary()
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "error": "Model not loaded"
        }), 503

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main therapeutic chat endpoint
    
    Expected JSON payload:
    {
        "message": "User's concern or question",
        "use_history": true/false (optional, default: true),
        "temperature": 0.7 (optional, range: 0.1-1.0),
        "max_length": 512 (optional)
    }
    
    Response:
    {
        "response": "Therapeutic response",
        "conversation_stats": {...}
    }
    """
    if not chat_handler:
        return jsonify({
            "error": "Therapeutic model not initialized. Please wait..."
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' in request body"
            }), 400
        
        user_message = data['message']
        
        # Validate message
        if not user_message.strip():
            return jsonify({
                "error": "Empty message"
            }), 400
        
        # Get optional parameters with therapeutic defaults
        use_history = data.get('use_history', True)
        temperature = float(data.get('temperature', 0.7))
        max_length = int(data.get('max_length', 512))
        top_p = float(data.get('top_p', 0.9))
        top_k = int(data.get('top_k', 50))
        repetition_penalty = float(data.get('repetition_penalty', 1.1))
        
        # Validate parameters
        temperature = max(0.1, min(1.0, temperature))
        max_length = max(50, min(1024, max_length))
        
        print(f"\n{'='*50}")
        print(f"User: {user_message[:100]}...")
        print(f"{'='*50}")
        
        # Generate therapeutic response
        response = chat_handler.generate_response(
            user_message=user_message,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_history=use_history,
            repetition_penalty=repetition_penalty
        )
        
        print(f"Counselor: {response[:100]}...")
        print(f"{'='*50}\n")
        
        return jsonify({
            "response": response,
            "conversation_stats": chat_handler.get_conversation_summary(),
            "timestamp": data.get('timestamp', None)
        }), 200
        
    except Exception as e:
        print(f"✗ Error processing chat request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Error processing request: {str(e)}"
        }), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history - start new counseling session"""
    if not chat_handler:
        return jsonify({
            "error": "Model not initialized"
        }), 503
    
    try:
        chat_handler.clear_history()
        return jsonify({
            "message": "Conversation history cleared. Starting new session.",
            "conversation_stats": chat_handler.get_conversation_summary()
        }), 200
    except Exception as e:
        return jsonify({
            "error": f"Error clearing history: {str(e)}"
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get full conversation history"""
    if not chat_handler:
        return jsonify({
            "error": "Model not initialized"
        }), 503
    
    return jsonify({
        "history": chat_handler.get_history(),
        "stats": chat_handler.get_conversation_summary()
    }), 200

@app.route('/api/history', methods=['POST'])
def set_history():
    """Set conversation history (for session restoration)"""
    if not chat_handler:
        return jsonify({
            "error": "Model not initialized"
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'history' not in data:
            return jsonify({
                "error": "Missing 'history' in request body"
            }), 400
        
        chat_handler.set_history(data['history'])
        return jsonify({
            "message": "History restored successfully",
            "conversation_stats": chat_handler.get_conversation_summary()
        }), 200
    except Exception as e:
        return jsonify({
            "error": f"Error updating history: {str(e)}"
        }), 500

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get model and application information"""
    if not model_loader:
        return jsonify({
            "error": "Model not initialized"
        }), 503
    
    return jsonify({
        "model_info": model_loader.get_model_info(),
        "app_info": {
            "name": "Gemma 3n Therapeutic Chatbot",
            "version": "1.0",
            "flask_app_path": FLASK_APP_PATH,
            "model_type": "Mental Health Counseling"
        }
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(TEMPLATE_PATH, exist_ok=True)
    os.makedirs(STATIC_PATH, exist_ok=True)
    
    # Initialize model before starting server
    if initialize_model():
        # Run Flask app
        print("Starting Flask server...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to True for development
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
    else:
        print("\n✗ Failed to initialize therapeutic model. Exiting.")
        sys.exit(1)

from process_rag import ProcessRAG
from flask import Flask, request, jsonify

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variable for ProcessRAG instance
rag_processor = None

def initialize():
    """
    Initialization method to create an instance of ProcessRAG.
    This runs once when the Flask app starts.
    """
    global rag_processor
    print("Initializing the application...")

    rag_processor = ProcessRAG()
    
    print("Initialization completed successfully.")

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate an answer based on the user's question.
    """
    global rag_processor

    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "Question is required"}), 400
    try:
        print(question)
        answer = rag_processor.process_prompt(question)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the initialization method once before the server starts
    initialize()
    app.run(host="0.0.0.0", port=5000)







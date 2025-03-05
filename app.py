import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from models.chat_history import db, ChatMessage
import uuid

from config import DEBUG, DOCUMENT_STORE_PATH
from rag.document_processor import DocumentProcessor
from rag.rag_pipeline import RAGPipeline
from rag.output_formatter import OutputFormatter
from agents.agent_factory import AgentFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Ensure document store directory exists
os.makedirs(DOCUMENT_STORE_PATH, exist_ok=True)

# Configure upload settings
app.config['UPLOAD_FOLDER'] = DOCUMENT_STORE_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'md'}

# Initialize components
document_processor = DocumentProcessor()
rag_pipeline = RAGPipeline()

# Initialize retrievers for both namespaces
rag_pipeline.retriever.retrieve(query="", namespace="inventory")  # Initialize inventory retriever
rag_pipeline.retriever.retrieve(query="", namespace="transport")  # Initialize transport retriever

output_formatter = OutputFormatter()

# Initialize the database
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document uploads."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        try:
            # Save the file
            file_path = document_processor.save_document(file, filename)
            
            # Process and ingest the document
            chunk_ids = rag_pipeline.ingest_document(file_path)
            
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'filename': filename,
                'chunks': len(chunk_ids)
            }), 200
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/query', methods=['POST'])
def query():
    """Handle RAG queries."""
    try:
        data = request.json
        query = data.get('query')
        chat_id = data.get('chat_id')

        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Generate response using RAG pipeline with multi-namespace query
        rag_response = rag_pipeline.query_multi_namespace(query)
        
        # Format the response
        formatted_response = output_formatter.format_response(rag_response['response'])
        
        # Get the chat title from existing messages if available
        title = None
        if chat_id:
            existing_message = ChatMessage.query.filter_by(chat_id=chat_id).first()
            if existing_message:
                title = existing_message.title
        else:
            # Use truncated query as title for new chat
            title = query[:50] + ('...' if len(query) > 50 else '')

        # Create new message
        message = ChatMessage(
            chat_id=chat_id or str(uuid.uuid4()),
            role='assistant',
            content=rag_response['response'],
            title=title
        )
        
        # Set the formatted response
        message.set_formatted_response(formatted_response.to_dict())
        
        # Save to database
        db.session.add(message)
        db.session.commit()

        return jsonify({
            'formatted_response': formatted_response.to_dict(),
            'message': message.to_dict(),
            'inventory_documents': rag_response.get('inventory_documents', []),
            'transport_documents': rag_response.get('transport_documents', [])
        })

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/agent', methods=['POST'])
def agent_query():
    """Handle agent-based queries."""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query_text = data['query']
    agent_type = data.get('agent_type', 'research')  # Default to research agent
    
    try:
        # Create and run the appropriate agent
        agent = AgentFactory.create_agent(agent_type, rag_pipeline)
        result = agent.run(query_text)
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error processing agent query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents."""
    try:
        documents = []
        for filename in os.listdir(DOCUMENT_STORE_PATH):
            if allowed_file(filename):
                file_path = os.path.join(DOCUMENT_STORE_PATH, filename)
                file_size = os.path.getsize(file_path)
                documents.append({
                    'filename': filename,
                    'size': file_size,
                    'uploaded_at': os.path.getctime(file_path)
                })
        
        return jsonify({'documents': documents}), 200
    
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<filename>', methods=['GET'])
def download_document(filename):
    """Download a document."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/chat_history', methods=['GET', 'POST'])
def chat_history():
    if request.method == 'GET':
        # Get all unique chats with their latest title and message count
        chats = db.session.query(
            ChatMessage.chat_id,
            db.func.max(ChatMessage.title).label('title'),
            db.func.min(ChatMessage.timestamp).label('created_at'),
            db.func.count(ChatMessage.id).label('message_count')
        ).group_by(ChatMessage.chat_id).order_by(db.text('created_at DESC')).all()
        
        return jsonify({
            'chats': [{
                'id': chat.chat_id,
                'title': chat.title or 'New Chat',
                'created_at': chat[2].isoformat() if chat[2] else None,
                'message_count': chat.message_count
            } for chat in chats]
        })
    
    elif request.method == 'POST':
        data = request.json
        chat_id = data.get('chat_id')
        title = data.get('title')
        
        if not chat_id or not title:
            return jsonify({'error': 'Missing chat_id or title'}), 400
        
        # Update the title for all messages in this chat
        messages = ChatMessage.query.filter_by(chat_id=chat_id).all()
        if not messages:
            return jsonify({'error': 'Chat not found'}), 404
            
        for message in messages:
            message.title = title
        
        db.session.commit()
        return jsonify({'success': True})

@app.route('/chat_history/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat and all its messages."""
    try:
        # Delete all messages for this chat
        messages = ChatMessage.query.filter_by(chat_id=chat_id).all()
        for message in messages:
            db.session.delete(message)
        
        db.session.commit()
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/messages', methods=['POST'])
def save_message():
    data = request.json
    chat_id = data.get('chat_id') or str(uuid.uuid4())
    role = data.get('role')
    content = data.get('content')
    title = data.get('title')  # Optional title for the chat
    formatted_response = data.get('formatted_response')  # Optional formatted response
    
    if not role or not content:
        return jsonify({'error': 'Missing role or content'}), 400
    
    message = ChatMessage(
        chat_id=chat_id,
        role=role,
        content=content,
        title=title  # Will be None if not provided
    )
    
    # Set formatted response if provided
    if formatted_response:
        message.set_formatted_response(formatted_response)
    
    # If this is the first message in a chat and no title is provided, use the content
    if not title:
        existing_messages = ChatMessage.query.filter_by(chat_id=chat_id).first()
        if not existing_messages and role == 'user':
            truncated_content = content[:50] + ('...' if len(content) > 50 else '')
            message.title = truncated_content
    
    db.session.add(message)
    db.session.commit()
    
    return jsonify(message.to_dict())

@app.route('/messages/<chat_id>', methods=['GET'])
def get_messages(chat_id):
    messages = ChatMessage.query.filter_by(chat_id=chat_id).order_by(ChatMessage.timestamp).all()
    return jsonify({
        'messages': [message.to_dict() for message in messages]
    })

@app.route('/chat_history/clear', methods=['POST'])
def clear_chat_history():
    """Clear all chat history."""
    try:
        # Clear chat history in RAG pipeline
        rag_pipeline.clear_chat_history()
        
        # Clear chat history from database
        ChatMessage.query.delete()
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Chat history cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error clearing chat history: {str(e)}'
        }), 500

@app.route('/chat_history/entry/<timestamp>', methods=['DELETE'])
def delete_chat_entry(timestamp):
    """Delete a specific chat entry by timestamp."""
    try:
        # Delete from RAG pipeline
        result = rag_pipeline.delete_chat_entry(timestamp)
        
        # Delete from database
        message = ChatMessage.query.filter_by(timestamp=timestamp).first()
        if message:
            db.session.delete(message)
            db.session.commit()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error deleting chat entry: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error deleting chat entry: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)

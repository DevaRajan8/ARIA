"""
ARIA Web Application
This module provides a web API interface for ARIA.
"""

from flask import Flask, request, jsonify, session
import time
import uuid
import logging
from functools import wraps

# Import ARIA modules
from config import config
from models import TherapeuticModel, AnalysisModel
from memory.vector_memory import VectorMemory
from memory.enhanced_memory import EnhancedMemory
from graph.conversation_graph import ConversationGraph
from graph.workflow import Workflow
from analyzers.personality_analyzer import PersonalityAnalyzer
from analyzers.therapeutic_analyzer import TherapeuticAnalyzer
from seal.seal_framework import SEALFramework
from tools.crisis_tool import CrisisTool
from tools.therapeutic_tool import TherapeuticTool
from utils.monitoring import Monitor

# Initialize Flask app
app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

# Configure app
app.config['DEBUG'] = config.get('api.debug', False)
app.config['HOST'] = config.get('api.host', 'localhost')
app.config['PORT'] = config.get('api.port', 5000)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not app.config['DEBUG'] else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.get('system.log_dir') + '/app.log'
)
logger = logging.getLogger('ARIA.App')

# Initialize ARIA components
monitor = Monitor()
therapeutic_model = TherapeuticModel('default')
analysis_model = AnalysisModel('default')
vector_memory = VectorMemory()
enhanced_memory = EnhancedMemory()
conversation_graph = ConversationGraph()
seal_framework = SEALFramework()
crisis_tool = CrisisTool()
therapeutic_tool = TherapeuticTool()

# Start monitoring
monitor.start_monitoring(interval=config.get('system.monitor_interval', 60))

# Authentication decorator
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        # Simple token validation (in production, use proper auth)
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthorized'}), 401
            
        # In a real app, validate the token here
        # For this example, we'll just check it exists
        token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(*args, **kwargs)
    return decorated

# Error handler
@app.errorhandler(Exception)
def handle_error(error):
    logger.exception(f"Unhandled exception: {str(error)}")
    monitor.log_event('error.unhandled', {'message': str(error)})
    return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'version': config.get('system.version'),
        'timestamp': time.time()
    })

# Session management endpoints
@app.route('/sessions', methods=['POST'])
@require_auth
def create_session():
    data = request.json
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
        
    # Create a new session
    session_id = str(uuid.uuid4())
    
    # Initialize session state
    therapeutic_tool.start_session(session_id)
    
    # Log event
    monitor.log_event('session.created', {
        'session_id': session_id,
        'user_id': user_id
    })
    
    return jsonify({
        'session_id': session_id,
        'created_at': time.time()
    })

# Message endpoint
@app.route('/sessions/<session_id>/messages', methods=['POST'])
@require_auth
def send_message(session_id):
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
        
    # Process the message
    try:
        # Check for crisis indicators
        crisis_analysis = crisis_tool.analyze_message(message)
        
        if crisis_analysis['should_intervene']:
            # Handle potential crisis
            response = crisis_tool.get_response(crisis_analysis)
            
            # Log crisis event
            monitor.log_event('crisis.detected', {
                'session_id': session_id,
                'risk_level': crisis_analysis['highest_risk'],
                'categories': crisis_analysis['categories']
            })
        else:
            # Normal therapeutic response
            input_data = {
                'message': message,
                'session_id': session_id,
                'timestamp': time.time()
            }
            
            # Generate response
            model_response = therapeutic_model.predict(input_data)
            
            # Apply SEAL framework to validate response
            seal_result = seal_framework.evaluate_response(
                model_response['response'],
                {'message': message, 'session_id': session_id}
            )
            
            if not seal_result['passed']:
                # Response didn't pass SEAL checks
                response = {
                    'message': 'I need to consider how to respond appropriately. Can you tell me more?',
                    'technique': 'clarification'
                }
                
                # Log SEAL rejection
                monitor.log_event('seal.rejected', {
                    'session_id': session_id,
                    'reasons': seal_result
                })
            else:
                response = {
                    'message': model_response['response'],
                    'technique': model_response['technique']
                }
                
        # Add message and response to conversation graph
        message_node_id = f"{session_id}-{time.time()}-user"
        response_node_id = f"{session_id}-{time.time()}-aria"
        
        conversation_graph.add_node(message_node_id, message, {'speaker': 'user'})
        conversation_graph.add_node(response_node_id, response['message'], {'speaker': 'aria'})
        conversation_graph.add_edge(message_node_id, response_node_id, 'response')
        
        # Log event
        monitor.log_event('message.processed', {
            'session_id': session_id,
            'technique': response.get('technique')
        })
        
        return jsonify({
            'message': response['message'],
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.exception(f"Error processing message: {str(e)}")
        monitor.log_event('error.message_processing', {'message': str(e)})
        return jsonify({'error': 'Failed to process message'}), 500

# Analysis endpoint
@app.route('/sessions/<session_id>/analyze', methods=['POST'])
@require_auth
def analyze_session(session_id):
    data = request.json
    analysis_types = data.get('types', ['personality', 'therapeutic'])
    
    try:
        results = {}
        
        if 'personality' in analysis_types:
            analyzer = PersonalityAnalyzer()
            # In a real app, load conversation history for the analyzer
            results['personality'] = analyzer.get_insights()
            
        if 'therapeutic' in analysis_types:
            analyzer = TherapeuticAnalyzer()
            # In a real app, load conversation history for the analyzer
            results['therapeutic'] = analyzer.analyze_session(session_id)
            
        # Log event
        monitor.log_event('analysis.completed', {
            'session_id': session_id,
            'types': analysis_types
        })
        
        return jsonify(results)
        
    except Exception as e:
        logger.exception(f"Error analyzing session: {str(e)}")
        monitor.log_event('error.analysis', {'message': str(e)})
        return jsonify({'error': 'Failed to analyze session'}), 500

# Main entry point
if __name__ == '__main__':
    logger.info(f"Starting ARIA API on {app.config['HOST']}:{app.config['PORT']}")
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )

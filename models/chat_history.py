from datetime import datetime
import json
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True)
    chat_id = Column(String(50), nullable=False)  # UUID for grouping messages in a chat
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    title = Column(String(200))  # Auto-generated title for the chat
    formatted_response = Column(Text)  # Store formatted response as JSON string

    def to_dict(self):
        return {
            'id': self.id,
            'chat_id': self.chat_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'title': self.title,
            'formatted_response': json.loads(self.formatted_response) if self.formatted_response else None
        }

    def set_formatted_response(self, response_dict):
        """Set the formatted response by converting dict to JSON string."""
        if response_dict is not None:
            self.formatted_response = json.dumps(response_dict)
        else:
            self.formatted_response = None 
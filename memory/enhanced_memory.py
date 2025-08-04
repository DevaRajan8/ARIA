# aria_system/memory/enhanced_memory.py
import uuid
import networkx as nx
from datetime import datetime
from typing import List, Dict, Any
from zep_cloud.client import Zep
from zep_cloud import Message
from config import ZEP_API_KEY, logger

class EnhancedMemorySystem:
    def __init__(self, zep_api_key: str = ZEP_API_KEY):
        self.zep = Zep(api_key=zep_api_key)
        self.relationship_graph = nx.Graph()
        self.memory_cache = {}
        logger.info("Enhanced memory system initialized")

    def get_or_create_session(self, user_id: str) -> str:
        session_id = f"aria_session_{user_id}_{uuid.uuid4().hex[:8]}"
        try:
            self.zep.memory.add_session(session_id=session_id, user_id=user_id)
            self.relationship_graph.add_node(user_id, type="user")
            self.relationship_graph.add_node(session_id, type="session")
            self.relationship_graph.add_edge(user_id, session_id, relationship="owns")
            return session_id
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return session_id

    def add_conversation(self, session_id: str, user_message: str, 
                        assistant_response: str, metadata: Dict[str, Any] = None):
        try:
            messages = [
                Message(content=user_message, role_type="user", metadata=metadata or {}),
                Message(
                    content=assistant_response, 
                    role_type="assistant",
                    metadata={
                        **(metadata or {}),
                        "timestamp": datetime.now().isoformat(),
                        "confidence": metadata.get("confidence", 0.0) if metadata else 0.0
                    }
                )
            ]
            self.zep.memory.add(session_id=session_id, messages=messages)
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
            self.relationship_graph.add_node(message_id, type="message")
            self.relationship_graph.add_edge(session_id, message_id, relationship="contains")
        except Exception as e:
            logger.error(f"Memory storage error: {e}")

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        try:
            response = self.zep.memory.get_session_messages(session_id=session_id, limit=limit)
            messages = response.messages if hasattr(response, 'messages') else response
            history = []
            for msg in messages:
                role = "user" if getattr(msg, 'role_type', '') == "user" else "assistant"
                content = getattr(msg, 'content', '')
                metadata = getattr(msg, 'metadata', {})
                history.append({"role": role, "content": content, "metadata": metadata})
            return history
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []

    def get_enhanced_context(self, session_id: str, user_id: str, 
                           current_query: str) -> Dict[str, Any]:
        recent_history = self.get_conversation_history(session_id, limit=10)
        user_patterns = self.analyze_cross_session_patterns(user_id)
        semantic_context = self._get_semantic_context(user_id, current_query)
        return {
            "recent_history": recent_history,
            "user_patterns": user_patterns,
            "semantic_context": semantic_context,
            "relationship_context": self._get_relationship_context(user_id)
        }

    def analyze_cross_session_patterns(self, user_id: str) -> Dict[str, Any]:
        user_sessions = [
            node for node in self.relationship_graph.nodes()
            if (self.relationship_graph.nodes[node].get('type') == 'session' and
                any(neighbor == user_id for neighbor in self.relationship_graph.neighbors(node)))
        ]
        patterns = {
            "total_sessions": len(user_sessions),
            "conversation_topics": [],
            "emotional_progression": [],
            "interaction_frequency": {},
            "preferred_conversation_times": []
        }
        for session_id in user_sessions:
            history = self.get_conversation_history(session_id, limit=50)
            for msg in history:
                if msg["role"] == "user":
                    emotional_indicators = self._extract_emotional_indicators(msg["content"])
                    patterns["emotional_progression"].extend(emotional_indicators)
        return patterns

    def _get_semantic_context(self, user_id: str, query: str) -> Dict[str, Any]:
        return {
            "similar_queries": [],
            "context_strength": 0.0,
            "relevant_memories": []
        }

    def _get_relationship_context(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.relationship_graph:
            return {"connections": 0, "relationship_strength": 0.0}
        connections = list(self.relationship_graph.neighbors(user_id))
        return {
            "connections": len(connections),
            "relationship_strength": len(connections) / 10.0,
            "connection_types": [
                self.relationship_graph.nodes[conn].get('type', 'unknown')
                for conn in connections
            ]
        }

    def _extract_emotional_indicators(self, text: str) -> List[str]:
        emotions = []
        text_lower = text.lower()
        emotion_keywords = {
            "joy": ["happy", "excited", "wonderful", "great", "amazing"],
            "sadness": ["sad", "down", "depressed", "blue", "unhappy"],
            "anxiety": ["anxious", "worried", "nervous", "scared", "stressed"],
            "anger": ["angry", "frustrated", "mad", "annoyed", "irritated"],
            "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil"]
        }
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                emotions.append(emotion)
        return emotions
# aria_system/memory/vector_memory.py
import uuid
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict,Any
from models import PersonalityProfile
from config import logger, PINECONE_API_KEY

class VectorMemorySystem:
    def __init__(self, api_key: str = PINECONE_API_KEY):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.dimension = 384
        self.vectors = {}
        self.metadata = {}
        self.personality_vectors = {}
        logger.info("Vector memory system initialized")

    def encode_text(self, text: str) -> np.ndarray:
        return self.encoder.encode(text)

    def store_conversation_vector(self, user_id: str, conversation: str, 
                                metadata: Dict[str, Any]) -> str:
        vector = self.encode_text(conversation)
        vector_id = f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.vectors[vector_id] = vector
        self.metadata[vector_id] = {
            **metadata,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "type": "conversation"
        }
        return vector_id

    def search_similar_conversations(self, query: str, user_id: str, 
                                   top_k: int = 5) -> List[Dict]:
        query_vector = self.encode_text(query)
        similarities = []
        for vector_id, stored_vector in self.vectors.items():
            if self.metadata[vector_id].get("user_id") == user_id:
                similarity = cosine_similarity(
                    query_vector.reshape(1, -1), 
                    stored_vector.reshape(1, -1)
                )[0][0]
                similarities.append({
                    "vector_id": vector_id,
                    "similarity": similarity,
                    "metadata": self.metadata[vector_id]
                })
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def store_personality_vector(self, user_id: str, personality_profile: PersonalityProfile):
        trait_values = list(personality_profile.big_five.values())
        therapeutic_values = list(personality_profile.therapeutic_traits.values())
        combined_values = trait_values + therapeutic_values
        while len(combined_values) < self.dimension:
            combined_values.append(0.0)
        personality_vector = np.array(combined_values[:self.dimension])
        vector_id = f"personality_{user_id}"
        self.vectors[vector_id] = personality_vector
        self.metadata[vector_id] = {
            "user_id": user_id,
            "type": "personality",
            "confidence": personality_profile.confidence_score,
            "last_updated": personality_profile.last_updated.isoformat()
        }
        self.personality_vectors[user_id] = personality_vector

    def get_contextual_embeddings(self, user_id: str, query: str) -> Dict[str, Any]:
        similar_convs = self.search_similar_conversations(query, user_id, top_k=3)
        personality_context = self.personality_vectors.get(user_id)
        return {
            "query_vector": self.encode_text(query),
            "similar_conversations": similar_convs,
            "personality_context": personality_context,
            "context_strength": len(similar_convs)
        }
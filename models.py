from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

class ConversationMode(Enum):
    COMPANION = "companion"
    THERAPEUTIC = "therapeutic"
    CRISIS = "crisis"
    ASSESSMENT = "assessment"
    COACHING = "coaching"

class PersonalityTrait(Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    EMPATHY = "empathy"
    OPTIMISM = "optimism"
    EMOTIONAL_STABILITY = "emotional_stability"

class TherapeuticFramework(Enum):
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    MINDFULNESS = "mindfulness_based"
    SUPPORTIVE = "supportive_therapy"
    SOLUTION_FOCUSED = "solution_focused"

@dataclass
class PersonalityProfile:
    big_five: Dict[PersonalityTrait, float] = field(default_factory=dict)
    therapeutic_traits: Dict[str, float] = field(default_factory=dict)
    communication_preferences: Dict[str, Any] = field(default_factory=dict)
    emotional_patterns: List[str] = field(default_factory=list)
    adaptation_history: List[Dict] = field(default_factory=list)
    confidence_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TherapeuticAssessment:
    mood_score: float = 5.0
    anxiety_level: float = 5.0
    stress_indicators: List[str] = field(default_factory=list)
    coping_strategies: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    therapeutic_goals: List[str] = field(default_factory=list)
    progress_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SEALEdit:
    edit_id: str
    edit_type: str
    target_component: str
    synthetic_data: List[Dict[str, str]]
    hyperparameters: Dict[str, Any]
    effectiveness_score: float = 0.0
    implementation_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ARIAState:
    user_id: str
    session_id: str
    current_message: str
    conversation_history: List[Dict[str, str]]
    personality_profile: PersonalityProfile
    therapeutic_assessment: TherapeuticAssessment
    conversation_mode: ConversationMode
    context_vectors: List[float]
    memory_context: Dict[str, Any]
    seal_adaptations: List[SEALEdit]
    graph_context: Dict[str, Any]
    response: str = ""
    confidence: float = 0.0
    next_action: str = ""
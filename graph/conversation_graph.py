
import networkx as nx
from models import ARIAState, ConversationMode
import logging

logger = logging.getLogger(__name__)

class ConversationGraph:
    def __init__(self):
        self.flow_graph = nx.DiGraph()
        self.current_state = "initial"
        self.conversation_context = {}
        self._build_conversation_graph()
        logger.info("Conversation graph initialized")

    def _build_conversation_graph(self):
        states = [
            "initial", "greeting", "personality_assessment", "mood_check",
            "therapeutic_mode", "crisis_intervention", "companion_mode",
            "coaching_mode", "assessment_mode", "closure", "follow_up"
        ]
        for state in states:
            self.flow_graph.add_node(state)

        transitions = [
            ("initial", "greeting", {"condition": "first_message"}),
            ("greeting", "personality_assessment", {"condition": "new_user"}),
            ("greeting", "mood_check", {"condition": "returning_user"}),
            ("personality_assessment", "mood_check", {"condition": "assessment_complete"}),
            ("mood_check", "therapeutic_mode", {"condition": "distress_detected"}),
            ("mood_check", "companion_mode", {"condition": "neutral_mood"}),
            ("mood_check", "crisis_intervention", {"condition": "crisis_detected"}),
            ("therapeutic_mode", "coaching_mode", {"condition": "progress_made"}),
            ("therapeutic_mode", "crisis_intervention", {"condition": "crisis_escalation"}),
            ("companion_mode", "therapeutic_mode", {"condition": "support_needed"}),
            ("crisis_intervention", "therapeutic_mode", {"condition": "crisis_resolved"}),
            ("coaching_mode", "companion_mode", {"condition": "goal_achieved"}),
            ("assessment_mode", "therapeutic_mode", {"condition": "issues_identified"}),
            ("therapeutic_mode", "follow_up", {"condition": "session_end"}),
            ("companion_mode", "closure", {"condition": "conversation_end"}),
            ("coaching_mode", "closure", {"condition": "coaching_complete"})
        ]
        for source, target, data in transitions:
            self.flow_graph.add_edge(source, target, **data)

    def get_next_state(self, current_state: str, context: ARIAState) -> str:
        possible_states = list(self.flow_graph.successors(current_state))
        if not possible_states:
            return current_state
        for next_state in possible_states:
            edge_data = self.flow_graph[current_state][next_state]
            condition = edge_data.get("condition", "default")
            if self._evaluate_transition_condition(condition, context):
                return next_state
        return current_state

    def _evaluate_transition_condition(self, condition: str, context: ARIAState) -> bool:
        conditions = {
            "first_message": lambda: len(context.conversation_history) == 0,
            "new_user": lambda: context.personality_profile.confidence_score < 0.3,
            "returning_user": lambda: context.personality_profile.confidence_score >= 0.3,
            "assessment_complete": lambda: context.personality_profile.confidence_score >= 0.7,
            "distress_detected": lambda: (
                context.therapeutic_assessment.mood_score < 4.0 or 
                context.therapeutic_assessment.anxiety_level > 7.0
            ),
            "neutral_mood": lambda: (
                4.0 <= context.therapeutic_assessment.mood_score <= 7.0 and
                context.therapeutic_assessment.anxiety_level <= 7.0
            ),
            "crisis_detected": lambda: context.conversation_mode == ConversationMode.CRISIS,
            "progress_made": lambda: context.therapeutic_assessment.mood_score > 6.0,
            "crisis_escalation": lambda: len(context.therapeutic_assessment.risk_factors) > 0,
            "support_needed": lambda: any(
                indicator in context.current_message.lower() 
                for indicator in ["help", "support", "struggling", "difficult"]
            ),
            "crisis_resolved": lambda: (
                context.therapeutic_assessment.mood_score > 5.0 and
                len(context.therapeutic_assessment.risk_factors) == 0
            ),
            "goal_achieved": lambda: len(context.therapeutic_assessment.therapeutic_goals) > 0,
            "issues_identified": lambda: len(context.therapeutic_assessment.stress_indicators) > 0,
            "session_end": lambda: len(context.conversation_history) > 20,
            "conversation_end": lambda: "goodbye" in context.current_message.lower(),
            "coaching_complete": lambda: context.confidence > 0.8
        }
        condition_func = conditions.get(condition, lambda: True)
        return condition_func()

    def get_conversation_prompt(self, state: str, context: ARIAState) -> str:
        prompts = {
            "initial": "Welcome! I'm ARIA, your adaptive AI companion. How are you feeling today?",
            "greeting": f"Hello! It's nice to meet you. I'm here to provide support and companionship.",
            "personality_assessment": "I'd like to get to know you better. What kind of things do you enjoy talking about?",
            "mood_check": "How has your day been so far? I'm here to listen.",
            "therapeutic_mode": self._get_therapeutic_prompt(context),
            "crisis_intervention": "I can hear that you're going through a really difficult time right now. Your safety and wellbeing are important to me. Can you tell me more about how you're feeling?",
            "companion_mode": "I'm glad we can chat together. What's on your mind today?",
            "coaching_mode": "Let's work together on your goals. What would you like to focus on?",
            "assessment_mode": "I'd like to understand better how you've been feeling lately. Can you share more about your experiences?",
            "closure": "Thank you for sharing with me today. Remember, I'm always here when you need support.",
            "follow_up": "How are you feeling after our conversation? Is there anything else I can help you with?"
        }
        return prompts.get(state, "How can I help you today?")

    def _get_therapeutic_prompt(self, context: ARIAState) -> str:
        mood = context.therapeutic_assessment.mood_score
        anxiety = context.therapeutic_assessment.anxiety_level
        if mood < 3.0:
            return "I can sense you're going through a really tough time. Depression can feel overwhelming, but you don't have to face this alone. What's been weighing on you most heavily?"
        elif anxiety > 8.0:
            return "It sounds like you're experiencing a lot of anxiety right now. Let's try to work through this together. Can you tell me what's making you feel most anxious?"
        elif mood < 5.0:
            return "I notice you might be feeling down today. It's okay to have difficult days. What's been on your mind?"
        else:
            return "I'm here to support you. What would be most helpful for you right now?"

from typing import List, Tuple
from datetime import datetime
from models import TherapeuticAssessment
from config import logger

class TherapeuticAnalyzer:
    def __init__(self):
        self.mood_indicators = {
            "positive": ["happy", "great", "wonderful", "excellent", "fantastic", "amazing"],
            "neutral": ["okay", "fine", "normal", "usual", "average"],
            "negative": ["sad", "down", "terrible", "awful", "horrible", "depressed"]
        }
        self.anxiety_indicators = [
            "anxious", "worried", "panic", "nervous", "scared", "frightened",
            "overwhelmed", "stressed", "tense", "restless"
        ]
        self.crisis_indicators = [
            "suicide", "kill myself", "end it all", "can't go on", "hopeless",
            "want to die", "better off dead", "no point", "give up"
        ]
        self.coping_strategies = {
            "cognitive": ["reframe", "perspective", "think differently", "challenge thoughts"],
            "behavioral": ["exercise", "walk", "breathe", "relax", "activity"],
            "social": ["talk", "friends", "family", "support", "help"],
            "mindfulness": ["meditate", "mindful", "present", "aware", "focus"]
        }
        logger.info("Therapeutic analyzer initialized")

    def assess_mood(self, text: str) -> float:
        text_lower = text.lower()
        positive_count = sum(1 for word in self.mood_indicators["positive"] if word in text_lower)
        negative_count = sum(1 for word in self.mood_indicators["negative"] if word in text_lower)
        mood_adjustment = (positive_count - negative_count) * 0.5
        mood_score = 5.0 + mood_adjustment
        return max(1.0, min(10.0, mood_score))

    def assess_anxiety(self, text: str) -> float:
        text_lower = text.lower()
        anxiety_count = sum(1 for word in self.anxiety_indicators if word in text_lower)
        anxiety_score = 3.0 + (anxiety_count * 1.5)
        return max(1.0, min(10.0, anxiety_score))

    def detect_crisis(self, text: str) -> Tuple[bool, float]:
        text_lower = text.lower()
        crisis_count = sum(1 for indicator in self.crisis_indicators if indicator in text_lower)
        is_crisis = crisis_count > 0
        risk_level = min(crisis_count * 2.0, 10.0)
        return is_crisis, risk_level

    def identify_coping_strategies(self, text: str) -> List[str]:
        text_lower = text.lower()
        identified_strategies = []
        for category, strategies in self.coping_strategies.items():
            for strategy in strategies:
                if strategy in text_lower:
                    identified_strategies.append(f"{category}: {strategy}")
        return identified_strategies

    def update_therapeutic_assessment(self, current_assessment: TherapeuticAssessment,
                                    new_text: str) -> TherapeuticAssessment:
        new_mood = self.assess_mood(new_text)
        new_anxiety = self.assess_anxiety(new_text)
        is_crisis, risk_level = self.detect_crisis(new_text)
        coping_strategies = self.identify_coping_strategies(new_text)
        alpha = 0.3
        current_assessment.mood_score = (
            (1 - alpha) * current_assessment.mood_score + 
            alpha * new_mood
        )
        current_assessment.anxiety_level = (
            (1 - alpha) * current_assessment.anxiety_level + 
            alpha * new_anxiety
        )
        for strategy in coping_strategies:
            if strategy not in current_assessment.coping_strategies:
                current_assessment.coping_strategies.append(strategy)
        if is_crisis:
            current_assessment.risk_factors.append(f"Crisis indicators detected: {risk_level}")
        return current_assessment
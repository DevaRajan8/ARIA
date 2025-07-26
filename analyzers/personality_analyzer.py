import numpy as np
from typing import Dict
from models import PersonalityProfile, PersonalityTrait
from config import logger

class PersonalityAnalyzer:
    def __init__(self):
        self.personality_keywords = {
            PersonalityTrait.OPENNESS: ["creative", "curious", "open-minded", "imaginative", "artistic"],
            PersonalityTrait.CONSCIENTIOUSNESS: ["organized", "responsible", "reliable", "disciplined", "thorough"],
            PersonalityTrait.EXTRAVERSION: ["outgoing", "social", "talkative", "energetic", "assertive"],
            PersonalityTrait.AGREEABLENESS: ["kind", "cooperative", "trusting", "helpful", "sympathetic"],
            PersonalityTrait.NEUROTICISM: ["anxious", "worried", "stressed", "emotional", "sensitive"],
            PersonalityTrait.EMPATHY: ["understanding", "compassionate", "caring", "supportive"],
            PersonalityTrait.OPTIMISM: ["positive", "hopeful", "confident", "enthusiastic"]
        }
        self.communication_patterns = {
            "formal": ["please", "thank you", "would you", "could you", "appreciate"],
            "casual": ["hey", "yeah", "cool", "awesome", "no problem"],
            "emotional": ["feel", "emotion", "heart", "soul", "deeply"],
            "analytical": ["think", "analyze", "consider", "evaluate", "logical"]
        }
        logger.info("Personality analyzer initialized")

    def analyze_text_for_personality(self, text: str) -> Dict[PersonalityTrait, float]:
        text_lower = text.lower()
        word_count = len(text.split())
        trait_scores = {}
        for trait, keywords in self.personality_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = min(matches / max(len(keywords) * 0.3, 1), 1.0)
            score += np.random.normal(0, 0.1)
            score = max(0.0, min(1.0, score))
            trait_scores[trait] = score
        return trait_scores

    def detect_communication_style(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        style_scores = {}
        for style, keywords in self.communication_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            style_scores[style] = matches / len(keywords)
        return style_scores

    def update_personality_profile(self, current_profile: PersonalityProfile, 
                                 new_text: str) -> PersonalityProfile:
        new_traits = self.analyze_text_for_personality(new_text)
        new_comm_style = self.detect_communication_style(new_text)
        alpha = 0.1
        for trait, new_score in new_traits.items():
            if trait in current_profile.big_five:
                current_profile.big_five[trait] = (
                    (1 - alpha) * current_profile.big_five[trait] + 
                    alpha * new_score
                )
            else:
                current_profile.big_five[trait] = new_score
        for style, score in new_comm_style.items():
            if style in current_profile.communication_preferences:
                current_profile.communication_preferences[style] = (
                    (1 - alpha) * current_profile.communication_preferences[style] + 
                    alpha * score
                )
            else:
                current_profile.communication_preferences[style] = score
        current_profile.confidence_score = min(
            current_profile.confidence_score + 0.05, 
            1.0
        )
        current_profile.last_updated = datetime.now()
        return current_profile
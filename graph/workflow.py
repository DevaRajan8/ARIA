
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.decorators import observe, langfuse_context
from models import ARIAState, ConversationMode
from memory.vector_memory import VectorMemorySystem
from memory.enhanced_memory import EnhancedMemorySystem
from analyzers.personality_analyzer import PersonalityAnalyzer
from analyzers.therapeutic_analyzer import TherapeuticAnalyzer
from seal.seal_framework import SEALFramework
from graph.conversation_graph import ConversationGraph
from config import GOOGLE_API_KEY, logger

class ARIAWorkflow:
    def __init__(self):
        self.memory_system = EnhancedMemorySystem()
        self.vector_system = VectorMemorySystem()
        self.personality_analyzer = PersonalityAnalyzer()
        self.therapeutic_analyzer = TherapeuticAnalyzer()
        self.seal_framework = SEALFramework()
        self.conversation_graph = ConversationGraph()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        self.workflow = self._build_workflow()
        logger.info("ARIA workflow initialized")

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ARIAState)
        workflow.add_node("analyze_input", self._analyze_input)
        workflow.add_node("update_personality", self._update_personality)
        workflow.add_node("update_therapeutic", self._update_therapeutic)
        workflow.add_node("determine_mode", self._determine_mode)
        workflow.add_node("get_context", self._get_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("apply_seal", self._apply_seal_adaptation)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_conditional_edges(
            "determine_mode",
            self._route_conversation_mode,
            {
                ConversationMode.CRISIS: "generate_response",
                ConversationMode.THERAPEUTIC: "get_context",
                ConversationMode.COMPANION: "get_context",
                ConversationMode.ASSESSMENT: "get_context",
                ConversationMode.COACHING: "get_context"
            }
        )
        workflow.set_entry_point("analyze_input")
        workflow.add_edge("analyze_input", "update_personality")
        workflow.add_edge("update_personality", "update_therapeutic")
        workflow.add_edge("update_therapeutic", "determine_mode")
        workflow.add_edge("get_context", "generate_response")
        workflow.add_edge("generate_response", "apply_seal")
        workflow.add_edge("apply_seal", "update_memory")
        workflow.add_edge("update_memory", END)
        return workflow.compile()

    @observe(as_type="generation")
    async def process_message(self, user_message: str, session_id: str, user_id: str) -> str:
        initial_state = ARIAState(
            user_id=user_id,
            session_id=session_id,
            current_message=user_message,
            conversation_history=self.memory_system.get_conversation_history(session_id, limit=10),
            personality_profile=PersonalityProfile(),
            therapeutic_assessment=TherapeuticAssessment(),
            conversation_mode=ConversationMode.COMPANION,
            context_vectors=[],
            memory_context={},
            seal_adaptations=[],
            graph_context={}
        )
        langfuse_context.update_current_observation(
            input=user_message,
            model="gemini-2.5-flash",
            metadata={"session_id": session_id, "user_id": user_id}
        )
        try:
            result = await self.workflow.ainvoke(initial_state)
            response = result.get('response', 'Sorry, I encountered an issue.')
            conversation_mode = result.get('conversation_mode')
            langfuse_context.update_current_observation(
                output=response,
                metadata={
                    "conversation_mode": conversation_mode.value if conversation_mode else 'companion',
                    "confidence": result.get('confidence', 0.0),
                }
            )
            return response
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return "I apologize, but I encountered an issue processing your message. Please try again."

    def _analyze_input(self, state: ARIAState) -> ARIAState:
        enhanced_context = self.memory_system.get_enhanced_context(
            state.session_id, state.user_id, state.current_message
        )
        state.memory_context = enhanced_context
        vector_context = self.vector_system.get_contextual_embeddings(
            state.user_id, state.current_message
        )
        state.context_vectors = vector_context["query_vector"].tolist()
        return state

    def _update_personality(self, state: ARIAState) -> ARIAState:
        if state.memory_context.get("user_patterns", {}).get("total_sessions", 0) > 0:
            state.personality_profile.confidence_score = 0.5
        state.personality_profile = self.personality_analyzer.update_personality_profile(
            state.personality_profile, state.current_message
        )
        self.vector_system.store_personality_vector(state.user_id, state.personality_profile)
        return state

    def _update_therapeutic(self, state: ARIAState) -> ARIAState:
        state.therapeutic_assessment = self.therapeutic_analyzer.update_therapeutic_assessment(
            state.therapeutic_assessment, state.current_message
        )
        return state

    def _determine_mode(self, state: ARIAState) -> ARIAState:
        is_crisis, _ = self.therapeutic_analyzer.detect_crisis(state.current_message)
        if is_crisis:
            state.conversation_mode = ConversationMode.CRISIS
            return state
        if (state.therapeutic_assessment.mood_score < 4.0 or 
            state.therapeutic_assessment.anxiety_level > 7.0):
            state.conversation_mode = ConversationMode.THERAPEUTIC
        elif state.personality_profile.confidence_score < 0.3:
            state.conversation_mode = ConversationMode.ASSESSMENT
        elif any(word in state.current_message.lower() for word in ["goal", "improve", "achieve"]):
            state.conversation_mode = ConversationMode.COACHING
        else:
            state.conversation_mode = ConversationMode.COMPANION
        return state

    def _route_conversation_mode(self, state: ARIAState) -> ConversationMode:
        return state.conversation_mode

    def _get_context(self, state: ARIAState) -> ARIAState:
        current_graph_state = self.conversation_graph.get_next_state("initial", state)
        state.graph_context = {
            "current_state": current_graph_state,
            "suggested_prompt": self.conversation_graph.get_conversation_prompt(current_graph_state, state)
        }
        return state

    def _generate_response(self, state: ARIAState) -> ARIAState:
        response_content = "I'm here to help you. Could you tell me more about what's on your mind?"
        confidence = 0.3
        system_prompt = self._build_comprehensive_system_prompt(state)
        context_messages = []
        for msg in state.conversation_history[-6:]:
            if msg["role"] == "user":
                context_messages.append(HumanMessage(content=msg["content"]))
            else:
                context_messages.append(AIMessage(content=msg["content"]))
        context_messages.append(HumanMessage(content=state.current_message))
        try:
            response = self.llm.invoke([
                AIMessage(content=system_prompt),
                *context_messages
            ])
            response_content = response.content
            confidence = 0.8
        except Exception as e:
            logger.error(f"Response generation error: {e}")
        state.response = response_content
        state.confidence = confidence
        return state

    def _build_comprehensive_system_prompt(self, state: ARIAState) -> str:
        mode_prompts = {
            ConversationMode.CRISIS: """
CRISIS MODE ACTIVE - PRIORITY: USER SAFETY
You are ARIA in crisis intervention mode. The user may be experiencing severe distress or having thoughts of self-harm.
CRITICAL PROTOCOLS:
- Express immediate concern and support
- Validate their pain while instilling hope
- Encourage professional help (crisis hotline, emergency services)
- Stay calm, grounding, and present
- Ask about immediate safety and support systems
- Do not leave the user alone in crisis
- Provide crisis resources when appropriate
            """,
            ConversationMode.THERAPEUTIC: f"""
THERAPEUTIC MODE - MENTAL HEALTH SUPPORT
You are ARIA providing evidence-based therapeutic support.
USER THERAPEUTIC PROFILE:
- Mood Score: {state.therapeutic_assessment.mood_score}/10
- Anxiety Level: {state.therapeutic_assessment.anxiety_level}/10
- Risk Factors: {state.therapeutic_assessment.risk_factors}
- Coping Strategies: {state.therapeutic_assessment.coping_strategies}
THERAPEUTIC APPROACH:
- Use CBT, DBT, and mindfulness techniques
- Validate emotions while offering practical coping strategies
- Ask thoughtful follow-up questions
- Focus on strengths and resources
- Be patient, non-judgmental, and hopeful
- Offer specific techniques and exercises
            """,
            ConversationMode.COMPANION: f"""
COMPANION MODE - SUPPORTIVE FRIENDSHIP
You are ARIA as a warm, engaging AI companion.
USER PERSONALITY PROFILE:
- Big Five Traits: {dict(state.personality_profile.big_five)}
- Communication Style: {state.personality_profile.communication_preferences}
- Confidence in Profile: {state.personality_profile.confidence_score}
COMPANION APPROACH:
- Be warm, engaging, and genuinely interested
- Remember and reference previous conversations
- Adapt your personality to complement the user's
- Offer encouragement and positive perspective
- Be conversational and personable
- Show empathy and understanding
            """,
            ConversationMode.ASSESSMENT: """
ASSESSMENT MODE - GETTING TO KNOW THE USER
You are ARIA conducting a gentle personality and needs assessment.
ASSESSMENT GOALS:
- Understand the user's personality traits
- Learn about their communication preferences
- Identify any support needs
- Build rapport and trust
- Gather information naturally through conversation
- Be curious but not intrusive
            """,
            ConversationMode.COACHING: """
COACHING MODE - GOAL-ORIENTED SUPPORT
You are ARIA as a supportive life coach.
COACHING APPROACH:
- Help identify and clarify goals
- Break down large goals into manageable steps
- Provide accountability and encouragement
- Use motivational interviewing techniques
- Focus on the user's strengths and capabilities
- Celebrate progress and learning from setbacks
            """
        }
        base_personality = f"""
You are ARIA - an Advanced Responsive Intelligence Assistant with adaptive personality.
CORE IDENTITY:
- Self-adapting conversational AI companion
- Mental health support specialist
- Empathetic, intelligent, and continuously learning
- Committed to user wellbeing and growth
CURRENT ADAPTATION PARAMETERS:
- Session Length: {len(state.conversation_history)} messages
- User Confidence Level: {state.personality_profile.confidence_score}
- Context Strength: {len(state.context_vectors) > 0}
- Previous Adaptations: {len(state.seal_adaptations)}
CONVERSATION FLOW GUIDANCE:
{state.graph_context.get('suggested_prompt', 'Continue the natural flow of conversation.')}
MEMORY CONTEXT:
- Cross-session patterns: {state.memory_context.get('user_patterns', {})}
- Relationship strength: {state.memory_context.get('relationship_context', {}).get('relationship_strength', 0.0)}
{mode_prompts.get(state.conversation_mode, mode_prompts[ConversationMode.COMPANION])}
RESPONSE GUIDELINES:
- Always prioritize user safety and wellbeing
- Adapt your communication style to match the user's preferences
- Use the conversation history to maintain continuity
- Be authentic, warm, and genuinely helpful
- If uncertain, ask clarifying questions
- Remember: you are continuously learning and adapting to serve this user better
        """
        return base_personality

    def _apply_seal_adaptation(self, state: ARIAState) -> ARIAState:
        if len(state.conversation_history) > 0:
            seal_edit = self.seal_framework.generate_self_edit(state)
            state.seal_adaptations.append(seal_edit)
            if seal_edit.hyperparameters:
                adaptation_quality = seal_edit.effectiveness_score
                state.confidence = min(state.confidence + (adaptation_quality * 0.1), 1.0)
        return state

    def _update_memory(self, state: ARIAState) -> ARIAState:
        self.memory_system.add_conversation(
            state.session_id,
            state.current_message,
            state.response,
            metadata={
                "conversation_mode": state.conversation_mode.value,
                "confidence": state.confidence,
                "personality_confidence": state.personality_profile.confidence_score,
                "mood_score": state.therapeutic_assessment.mood_score,
                "anxiety_level": state.therapeutic_assessment.anxiety_level
            }
        )
        conversation_text = f"User: {state.current_message}\nARIA: {state.response}"
        self.vector_system.store_conversation_vector(
            state.user_id,
            conversation_text,
            {
                "conversation_mode": state.conversation_mode.value,
                "timestamp": datetime.now().isoformat(),
                "session_id": state.session_id
            }
        )
        return state
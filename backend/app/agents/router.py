from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END

from ..core.llm import get_groq_client
from ..models.property import PropertySearchRequest
from .property_agent import PropertyAgent
from .compare_agent import CompareAgent
from .faq_agent import FAQAgent


class ConversationState(TypedDict):
    query: str
    conversation_history: list[dict]
    intent: str
    filters: dict
    response: str
    sources: list[str]
    error: Optional[str]


class CopilotOrchestrator:
    def __init__(self):
        self.groq_client = get_groq_client()
        self.property_agent = PropertyAgent()
        self.compare_agent = CompareAgent()
        self.faq_agent = FAQAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(ConversationState)
        
        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("handle_property_search", self._handle_property_search)
        graph.add_node("handle_compare", self._handle_compare)
        graph.add_node("handle_legal_faq", self._handle_legal_faq)
        graph.add_node("handle_unknown", self._handle_unknown)
        
        # Set entry point
        graph.set_entry_point("classify_intent")
        
        # Add conditional edges
        graph.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "property_search": "handle_property_search",
                "property_compare": "handle_compare",
                "legal_faq": "handle_legal_faq",
                "unknown": "handle_unknown",
            }
        )
        
        # Add edges to END
        graph.add_edge("handle_property_search", END)
        graph.add_edge("handle_compare", END)
        graph.add_edge("handle_legal_faq", END)
        graph.add_edge("handle_unknown", END)
        
        return graph.compile()

    def _classify_intent(self, state: ConversationState) -> ConversationState:
        """Classify the user's query intent."""
        prompt = (
            "Classify this query into exactly ONE category: property_search, property_compare, legal_faq, or unknown. "
            "Respond with ONLY the category name, nothing else.\n\n"
            f"Query: {state['query']}"
        )
        
        try:
            intent = self.groq_client.classify(prompt).strip().lower()
            
            # Validate intent
            valid_intents = ["property_search", "property_compare", "legal_faq", "unknown"]
            if intent not in valid_intents:
                intent = "unknown"
            
            state["intent"] = intent
        except Exception as e:
            state["intent"] = "unknown"
            state["error"] = str(e)
        
        return state

    def _route_intent(self, state: ConversationState) -> str:
        """Route to the appropriate handler based on intent."""
        return state["intent"]

    def _handle_property_search(self, state: ConversationState) -> ConversationState:
        """Handle property search queries."""
        try:
            # Create PropertySearchRequest
            request = PropertySearchRequest(
                query=state["query"],
                city=state["filters"].get("city"),
                min_price_lakhs=state["filters"].get("min_price_lakhs"),
                max_price_lakhs=state["filters"].get("max_price_lakhs"),
                bhk=state["filters"].get("bhk"),
                property_type=state["filters"].get("property_type"),
                top_k=state["filters"].get("top_k", 5),
            )
            
            # Call PropertyAgent
            response = self.property_agent.search(request)
            state["response"] = response.llm_summary
            state["sources"] = []
            
        except Exception as e:
            state["response"] = f"Error during property search: {str(e)}"
            state["error"] = str(e)
        
        return state

    def _handle_compare(self, state: ConversationState) -> ConversationState:
        """Handle property comparison queries."""
        try:
            # Try to extract property IDs from query or conversation history
            property_ids = self._extract_property_ids(state)
            
            if not property_ids or len(property_ids) < 2:
                # If no clear IDs, search for properties first
                search_request = PropertySearchRequest(query=state["query"], top_k=3)
                search_response = self.property_agent.search(search_request)
                property_ids = [prop.id for prop in search_response.results[:2]]
            
            if len(property_ids) < 2:
                state["response"] = "I need at least 2 properties to compare. Please specify the properties you want to compare."
                return state
            
            # Call CompareAgent
            comparison = self.compare_agent.compare(property_ids)
            state["response"] = comparison["comparison"]
            state["sources"] = []
            
        except Exception as e:
            state["response"] = f"Error during property comparison: {str(e)}"
            state["error"] = str(e)
        
        return state

    def _extract_property_ids(self, state: ConversationState) -> list[str]:
        """Extract property IDs from query or conversation history."""
        # Simple extraction - look for patterns like "property_123" or numeric IDs
        import re
        
        # Check query
        ids = re.findall(r'(?:property_)?(\d+)', state["query"])
        
        # Check conversation history
        for msg in state.get("conversation_history", []):
            if msg.get("role") == "assistant":
                found = re.findall(r'(?:property_)?(\d+)', msg.get("content", ""))
                ids.extend(found)
        
        # Convert to property_id format
        property_ids = [f"property_{id}" for id in ids[:4]]
        return property_ids

    def _handle_legal_faq(self, state: ConversationState) -> ConversationState:
        """Handle legal FAQ queries."""
        try:
            # Call FAQAgent
            answer = self.faq_agent.answer(state["query"])
            state["response"] = answer["answer"]
            state["sources"] = answer["sources"]
            
        except Exception as e:
            state["response"] = f"Error during FAQ answering: {str(e)}"
            state["error"] = str(e)
        
        return state

    def _handle_unknown(self, state: ConversationState) -> ConversationState:
        """Handle unknown intents."""
        state["response"] = (
            "I can help you: search properties, compare listings, or answer questions about "
            "Indian real estate laws (RERA, stamp duty, taxes, loans). What would you like?"
        )
        state["sources"] = []
        return state

    def invoke(
        self,
        query: str,
        conversation_history: Optional[list[dict]] = None,
        filters: Optional[dict] = None,
    ) -> dict:
        """Invoke the orchestrator with a user query."""
        if conversation_history is None:
            conversation_history = []
        if filters is None:
            filters = {}
        
        # Create initial state
        initial_state: ConversationState = {
            "query": query,
            "conversation_history": conversation_history,
            "intent": "",
            "filters": filters,
            "response": "",
            "sources": [],
            "error": None,
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return formatted response
        return {
            "response": final_state["response"],
            "intent": final_state["intent"],
            "sources": final_state.get("sources", []),
            "error": final_state.get("error"),
        }

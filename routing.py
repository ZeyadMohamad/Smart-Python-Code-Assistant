import os
import json
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOpenAI
# Keep your RAG hook available; used in generation path when helpful
from rag_pipeline import RAGPipeline

load_dotenv("main.env")

# ---------- State ----------

class AssistantState(TypedDict):
    user_input: str
    intent: str
    retrieved_examples: List[Dict[str, Any]]
    generated_response: str
    uploaded_files: List[Dict[str, str]]  # [{filename, text}]
    conversation_history: Annotated[list, add_messages]


# ---------- Intent Classifier ----------

class LLMIntentClassifier:
    """
    LLM-based intent classification (OpenRouter via LangChain ChatOpenAI).
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="openai/gpt-oss-20b:free",
            temperature=0.1,
            max_tokens=100,
        )

    def classify_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Classify the prompt into one of:
          - generate  (user wants to add/modify/produce code)
          - explain   (user wants explanation of code/files/concepts)
          - debug     (user wants help fixing errors, tracebacks, failing tests)
          - unsupported
        Return JSON: {"task": "<...>", "user_input": "<original>"}
        """
        prompt = f"""
You are an intent classifier for a Python code assistant.

Decide the task from ONLY the user's message (files may be attached as context later):
- "generate": add/modify/produce code (e.g., "add a function", "refactor", "write tests").
- "explain": explain a codebase/file/concept ("explain", "what does this do").
- "debug": fix errors/tracebacks/failing tests; user mentions "error", "traceback", "bug", "fails".
- "unsupported": unrelated to Python or unclear.

Respond ONLY with compact JSON exactly like:
{{"task":"<generate|explain|debug|unsupported>","user_input":"{user_input}"}}

Input: {user_input}
JSON:
""".strip()

        try:
            res = self.llm.invoke(prompt)
            content = (res.content or "").strip()
            if "{" in content:
                s = content.find("{")
                e = content.rfind("}") + 1
                return json.loads(content[s:e])
        except Exception as e:
            print(f"Intent classification error: {e}")

        # Fallback rules
        lower = user_input.lower()
        if any(k in lower for k in ["traceback", "exception", "error", "failing", "bug", "stack trace"]):
            return {"task": "debug", "user_input": user_input}
        if any(k in lower for k in ["explain", "what is", "what does", "how does"]):
            return {"task": "explain", "user_input": user_input}
        return {"task": "generate", "user_input": user_input}


# ---------- Main Assistant via LangGraph ----------

class LangGraphCodeAssistant:
    def __init__(self):
        self.intent_classifier = LLMIntentClassifier()
        self.rag_pipeline = RAGPipeline()

        # LLMs
        self.code_llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="deepseek/deepseek-chat-v3-0324:free",
            temperature=0.2,
            max_tokens=1024,
            request_timeout=60,
            max_retries=1                                                                      
        )
        self.explain_llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="deepseek/deepseek-chat-v3-0324:free",
            temperature=0.2,
            max_tokens=600,
            request_timeout=50,
            max_retries=1
        )
        self.debug_llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="tngtech/deepseek-r1t-chimera:free",
            temperature=0.2,
            max_tokens=700,
            request_timeout=50,
            max_retries=1
        )

        self.build_graph()

    # ---------- Nodes ----------

    def _format_files_for_context(self, files: List[Dict[str, str]], max_chars: int = 24000) -> str:
        """Serialize uploaded files into a compact, LLM-friendly section."""
        parts = []
        used = 0
        for f in files or []:
            name = f.get("filename", "uploaded_file.py")
            text = f.get("text", "")
            chunk = f"### FILE: {name}\n{text}\n"
            if used + len(chunk) > max_chars:
                remaining = max_chars - used
                if remaining > 200:
                    chunk = f"### FILE: {name}\n{text[:remaining]}\n... [truncated]\n"
                    parts.append(chunk)
                break
            parts.append(chunk)
            used += len(chunk)
        return "\n".join(parts) if parts else "(no attached files)"

    def unsupported_intent_node(self, state: AssistantState):
        state["generated_response"] = (
            "Sorry, I can only generate, explain, or debug Python code. "
            "Try: 'Add a function to ...', 'Explain this file ...', or 'Debug this error ...'."
        )
        return state

    def classify_intent_node(self, state: AssistantState):
        try:
            result = self.intent_classifier.classify_intent(state["user_input"])
            state["intent"] = result["task"]
            print(f"Intent classified as: {state['intent']}")
        except Exception as e:
            print(f"Intent classification error: {e}")
            state["intent"] = "generate"
        return state

    def retrieve_examples_node(self, state: AssistantState):
        try:
            examples = self.rag_pipeline.retrieve_examples(state["user_input"])
            state["retrieved_examples"] = examples or []
        except Exception as e:
            print(f"Retrieval error: {e}")
            state["retrieved_examples"] = []
        return state

    def generate_code_node(self, state: AssistantState):
        try:
            examples_text = ""
            for ex in state["retrieved_examples"]:
                examples_text += f"Example: {ex['prompt']}\nSolution: {ex['solution']}\n\n"

            files_ctx = self._format_files_for_context(state.get("uploaded_files", []))
            prompt = f"""You are a senior Python engineer.

User request:
{state['user_input']}

Attached files (use them if relevant; modify or add code as requested):
{files_ctx}

Retrieved examples (may be empty):
{examples_text}

Task: If the user asks to add/modify code in the attached files, propose a minimal patch.
Return:
1) Short plan
2) Unified diff (preferred) or corrected snippet
3) Notes/assumptions
"""
            res = self.code_llm.invoke(prompt)
            content = (res.content or "").strip()
            if not content.startswith("```"):
                content = f"```markdown\n{content}\n```"
            state["generated_response"] = content
        except Exception as e:
            state["generated_response"] = f"Error generating code: {e}"
        return state

    def explain_code_node(self, state: AssistantState):
        try:
            files_ctx = self._format_files_for_context(state.get("uploaded_files", []))
            prompt = f"""You are a Python tutor.

User question:
{state['user_input']}

If files are provided, explain *those files* in context of the question:
{files_ctx}

Provide:
- Clear explanation
- Key functions/classes and their roles
- Complexity/edge cases
- Suggestions for improvement (brief)
"""
            res = self.explain_llm.invoke(prompt)
            state["generated_response"] = (res.content or "").strip()
        except Exception as e:
            state["generated_response"] = f"Error explaining: {e}"
        return state

    def debug_file_node(self, state: AssistantState):
        try:
            files_ctx = self._format_files_for_context(state.get("uploaded_files", []))
            prompt = f"""You are a senior Python debugger.

User goal:
{state['user_input']}

Analyze the attached file(s), find likely issues, and propose a fix:
{files_ctx}

Return a compact, actionable report:

1) Summary (1–2 sentences)
2) Root cause analysis (bullets)
3) Code fix in python programming code:
```python
# patch here
4) quick checks
"""
            res = self.debug_llm.invoke(prompt)
            state["generated_response"] = (res.content or "").strip() or "No debug output."

        except Exception as e:
            state["generated_response"] = f"Error during debugging: {e}"

        return state
    
# ---------- Graph Definition ----------

    def build_graph(self):
        workflow = StateGraph(AssistantState)

        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("retrieve_examples", self.retrieve_examples_node)
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("explain_code", self.explain_code_node)
        workflow.add_node("debug_file", self.debug_file_node)
        workflow.add_node("unsupported_intent", self.unsupported_intent_node)

        workflow.set_entry_point("classify_intent")

        workflow.add_conditional_edges(
            "classify_intent",
            lambda s: s["intent"],
            {
                "generate": "retrieve_examples",
                "explain": "explain_code",
                "debug": "debug_file",
                "unsupported": "unsupported_intent",
            },
        )

        workflow.add_edge("retrieve_examples", "generate_code")
        workflow.add_edge("generate_code", END)
        workflow.add_edge("explain_code", END)
        workflow.add_edge("debug_file", END)
        workflow.add_edge("unsupported_intent", END)

        self.graph = workflow.compile()

# ---------- Public APIs ------------

    def process(self, user_input: str, uploaded_files: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Main entry: prompt is classified; files (if any) are provided as context to the routed node."""
        
        state: AssistantState = {
                "user_input": user_input,
                "intent": "",
                "retrieved_examples": [],
                "generated_response": "",
                "uploaded_files": uploaded_files or [],
                "conversation_history": [],
        }
        try:
            return self.graph.invoke(state)
        
        except Exception as e:

            print(f"Graph execution error: {e}")
            return {
            "user_input": user_input,
            "intent": "error",
            "retrieved_examples": [],
            "generated_response": f"Error processing request: {e}",
            "uploaded_files": uploaded_files or [],
            "conversation_history": [],
            }
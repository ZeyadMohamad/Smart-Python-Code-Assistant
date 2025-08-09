import gradio as gr
import os
import traceback
from typing import List, Dict, Any

from routing import LangGraphCodeAssistant
from rag_pipeline import RAGEvaluator  # keep your evaluator

# ---------------- Helpers ----------------

def _read_text(path: str) -> str:
    """Best-effort read of a text file; empty string for binary/unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

# ---------------- Init ----------------

def get_assistant():
    """Initialize the LangGraph assistant"""
    try:
        assistant = LangGraphCodeAssistant()
        print("Assistant initialized successfully!")
        return assistant
    except Exception as e:
        print(f"Error initializing assistant: {e}")
        traceback.print_exc()
        return None

print("Initializing LangGraph Code Assistant...")
assistant = get_assistant()

# ---------------- Handlers ----------------

def stage_files(new_files, staged: List[Dict[str, str]]):
    """
    Accumulate staged files (does NOT call the model).
    Returns updated staged list and a human-readable status string.
    """
    staged = staged or []
    added = []

    for f in (new_files or []):
        path = f if isinstance(f, str) else getattr(f, "name", None) or getattr(f, "orig_name", None)
        if not path:
            continue
        text = _read_text(path)
        added.append({"filename": os.path.basename(str(path)), "text": text})

    staged.extend(added)

    if not staged:
        status = "No files staged."
    else:
        status = "Currently staged files:\n" + "\n".join(
            f"- {x['filename']} ({len(x['text'])} chars)" for x in staged
        )
    return staged, status

def clear_all():
    """Clear chat, input, and staged files."""
    return [], "", []

def chatbot_with_files(user_input, history, staged_files):
    """
    Classify intent from the user's message and route; send staged files as context.
    Chatbot uses type='messages', so history expects list[dict(role, content)].
    """
    if not user_input.strip():
        return history, ""

    if not assistant:
        reply = "Sorry, the assistant failed to initialize. Please check your OPENROUTER_API_KEY."
    else:
        try:
            result = assistant.process(user_input, uploaded_files=staged_files or [])
            intent = result.get("intent", "unknown")
            response = result.get("generated_response", "No response generated.")
            num_files = len(staged_files or [])
            reply = f"**Intent:** {intent.capitalize()} | **Files:** {num_files}\n\n{response}"
        except Exception as e:
            reply = f"Error processing request: {str(e)}"
            print(f"Error in chatbot: {e}")
            traceback.print_exc()

    # Append as OpenAI-style messages
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    return history, ""

def evaluate_rag():
    """Run RAG evaluation on MBPP dataset (unchanged)"""
    if not assistant:
        return "Error: Assistant not initialized. Cannot run evaluation."

    try:
        evaluator = RAGEvaluator(assistant)
        print("Starting RAG evaluation...")

        results = evaluator.evaluate_on_mbpp(num_examples=10)

        if "error" in results:
            return f"Evaluation failed: {results['error']}"

        summary = f"""RAG Evaluation Results (10 MBPP Examples):

📊 Overall Metrics:
• Average Retrieval Quality: {results['avg_retrieval_quality']:.2f}
• Function Generation Rate: {results['function_generation_rate']:.1%}
• Total Examples Processed: {results['total_examples']}

📝 Sample Results:
"""
        for i, result in enumerate(results['detailed_results'][:3]):
            summary += f"""
Example {i+1}:
• Task ID: {result['task_id']}
• Intent: {result['intent']}
• Retrieved Examples: {result['num_retrieved']}
• Generated Function: {'✅' if result['has_function'] else '❌'}
• Prompt: {result['prompt'][:100]}...
"""
        return summary

    except Exception as e:
        error_msg = f"Error during evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def debug_single_example():
    """(Optional) Quick MBPP example debug — unchanged logic."""
    if not assistant:
        return "Error: Assistant not initialized."

    try:
        from datasets import load_dataset
        mbpp_test = load_dataset("mbpp", split="test")
        example = mbpp_test[0]

        result = assistant.process(example['text'])

        debug_info = f"""🔍 Debug Information for MBPP Example:

📝 Original Prompt: {example['text']}

🎯 Intent Classified: {result.get('intent', 'unknown')}

🔍 Retrieved Examples: {len(result.get('retrieved_examples', []))}
"""
        for i, ex in enumerate(result.get('retrieved_examples', [])[:2]):
            debug_info += f"""
Example {i+1} (Source: {ex['source']}):
• Task: {ex['prompt'][:80]}...
"""
        debug_info += f"""
🤖 Generated Response:
{result.get('generated_response', 'No response')}

✅ Expected Solution:
{example['code']}
"""
        return debug_info

    except Exception as e:
        return f"Debug error: {str(e)}"

# ---------------- UI ----------------

with gr.Blocks(title="Smart Python Code Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Smart Python Code Assistant")
    gr.Markdown("Upload file(s) ➜ type your message ➜ **Send**. I’ll classify your intent (generate/explain/debug) and use your files as context.")

    with gr.Row():
        with gr.Column(scale=2):
            # Use messages mode to avoid deprecation and to enable roles
            chatbot_ui = gr.Chatbot(label="Chat History", height=700, show_label=True, type="messages")

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="e.g. 'Explain this file', 'Debug the error in the uploaded file', 'Add a function to ...'",
                    lines=3,
                    scale=2,
                    show_label=False,
                )
                send_button = gr.Button("Send", variant="primary", scale=1)
                clear_button = gr.Button("Clear", variant="secondary", scale=1)

            # Staging uploader (does NOT call the model)
            file_uploader = gr.File(label="Add files", file_count="multiple", type="filepath")
            staged_files_state = gr.State([])  # [{filename, text}]
            staged_status = gr.Markdown("No files staged.")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 RAG Evaluation")
            gr.Markdown("Test the assistant on MBPP dataset")
            eval_button = gr.Button("Run RAG Evaluation", variant="primary")
            debug_button = gr.Button("Debug Single Example", variant="secondary")
            eval_output = gr.Textbox(label="Evaluation Results", lines=20, max_lines=25, show_label=True)

    # Events
    send_button.click(
        fn=chatbot_with_files,
        inputs=[user_input, chatbot_ui, staged_files_state],
        outputs=[chatbot_ui, user_input],
    )
    user_input.submit(
        fn=chatbot_with_files,
        inputs=[user_input, chatbot_ui, staged_files_state],
        outputs=[chatbot_ui, user_input],
    )
    clear_button.click(fn=clear_all, outputs=[chatbot_ui, user_input, staged_files_state]).then(
        lambda: "No files staged.", outputs=staged_status
    )

    # When files change, just stage them (don’t run LLM)
    file_uploader.change(
        fn=stage_files,
        inputs=[file_uploader, staged_files_state],
        outputs=[staged_files_state, staged_status],
    )

    eval_button.click(fn=evaluate_rag, outputs=eval_output)
    debug_button.click(fn=debug_single_example, outputs=eval_output)

if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch()

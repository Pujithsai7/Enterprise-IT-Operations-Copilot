import os
import streamlit as st
from validator import DiagnosticValidator

SYSTEM_PROMPT = """You are an Enterprise Network Diagnostic Assistant.

Your task is to determine whether the supplied evidence actually contains an error.

Strict Rules:
1. Do NOT assume an issue exists.
2. If no explicit error is present, clearly state:
   "No fault was detected from the supplied logs."
3. Recommend fixes ONLY if the retrieved documentation explicitly supports that diagnosis.
4. If evidence is insufficient, ask for additional logs.
5. Never invent failures from informational fields.
"""

class PlannerAgent:
    """
    Planner Agent: Integrates Diagnostic Validation Layer, formats evidence, and synthesizes cited fixes
    using OpenAI API or built-in Local Reasoning Engine.
    """
    def __init__(self):
        self.validator = DiagnosticValidator()

    def synthesize(self, query, doc_evidence, net_evidence, log_evidence, inc_evidence, chat_history=None, api_key=None, model_choice="Local Engine"):
        all_chunks = doc_evidence + net_evidence + log_evidence + inc_evidence
        
        # Pass retrieved chunks through Diagnostic Validation Layer
        validation_results = self.validator.validate(query, all_chunks)
        confidence_score = validation_results["confidence_score"]
        is_actual_error = validation_results["is_actual_error"]
        error_category = validation_results["error_category"]

        memory_str = ""
        if chat_history:
            history_snippets = [f"{m['role'].capitalize()}: {m['content'][:150]}" for m in chat_history[-4:]]
            memory_str = "\nRecent Conversation Memory:\n" + "\n".join(history_snippets)

        # Collect citations and snippets
        all_citations = []
        context_snippets = []
        for agent_name, ev_list in [("Documentation Agent", doc_evidence), ("Network Agent", net_evidence), ("Log Agent", log_evidence), ("Incident Agent", inc_evidence)]:
            for item in ev_list:
                cit = item.get('citation', '')
                if cit and cit not in all_citations:
                    all_citations.append(cit)
                context_snippets.append(f"[{agent_name} | {cit}]: {item['content'][:300]}")

        context_text = "\n".join(context_snippets) if context_snippets else "No specific document chunks retrieved."

        # Execute OpenAI API synthesis if API key is provided and model is OpenAI
        openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        if model_choice != "Local Engine" and openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                
                if not is_actual_error:
                    user_prompt_extra = f"""
VALIDATION LAYER AUDIT:
- Error Detected: NO (Operational Status Normal)
- Category: {error_category}
Follow Rule 2 strictly: State 'No fault was detected from the supplied logs.'
Follow Rule 4 strictly: Ask for additional logs if symptoms are observed.
Follow Rule 5 strictly: Never invent failures from informational fields.
"""
                else:
                    user_prompt_extra = f"""
VALIDATION LAYER AUDIT:
- Error Detected: YES
- Category: {error_category}
Follow Rule 3 strictly: Recommend fixes ONLY if the retrieved documentation explicitly supports that diagnosis.
"""

                user_prompt = f"""User Query: {query}
{memory_str}

Retrieved Document Context:
{context_text}

{user_prompt_extra}

Format response strictly as:
### Cause:
[Root cause summary OR "No fault was detected from the supplied logs."]

### Evidence:
- [Key evidence with source citations]

### Recommended Fix:
1. [Step 1: Specific action supported by documentation OR "Collect additional logs or describe specific observed symptoms."]
2. [Step 2]

### 📍 Source Citations:
- [List exact citations]
"""
                response = client.chat.completions.create(
                    model=model_choice if "gpt" in model_choice.lower() else "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                output_text = response.choices[0].message.content
                return (output_text, validation_results)
            except Exception as e:
                st.warning(f"OpenAI API call error ({str(e)}). Falling back to Local Reasoning Engine.")

        # Local Reasoning Engine Fallback (Zero external API dependency)
        if not is_actual_error:
            cause = "No fault was detected from the supplied logs."
            evidence_bullets = [
                "• The supplied telemetry and configurations show normal operational status with zero explicit error/fault indicators.",
                f"• Diagnostic Validation Layer confirmed status: `{error_category}`."
            ]
            rec = [
                "1. No corrective action required. The system telemetry appears operational.",
                "2. Collect additional logs or describe specific observed symptoms if an issue persists."
            ]
        else:
            evidence_bullets = []
            for name, items in [("Doc Agent", doc_evidence), ("Network Agent", net_evidence), ("Log Agent", log_evidence), ("Incident Agent", inc_evidence)]:
                if items:
                    evidence_bullets.append(f"• **{name}**: {items[0]['content'][:160]}... `{items[0].get('citation', '')}`")

            cause = f"Analysis of indexed company data indicates an operational fault ({error_category}) for query: '{query}'."
            rec = [
                f"1. Review diagnostic logs and configuration parameters in `{doc_evidence[0].get('citation', 'uploaded docs') if doc_evidence else 'system documentation'}`.",
                f"2. Apply standard resolution & interface configuration steps.",
                f"3. Verify link stability and monitor traffic metrics."
            ]
        
        cit_block = "\n".join([f"• `{c}`" for c in all_citations]) if all_citations else "• `[Uploaded Company Documents]`"
        
        output = f"### Cause:\n{cause}\n\n### Evidence:\n" + "\n".join(evidence_bullets) + "\n\n### Recommended Fix:\n" + "\n".join(rec) + f"\n\n### 📍 Source Citations:\n{cit_block}"
        return (output, validation_results)

from dotenv import load_dotenv
import os
import sys

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'streamlit'))

import streamlit as st
import json
import re

st.set_page_config(page_title="Business Assistant", page_icon="💬")
st.title("Internal Business Assistant")
st.caption("Look up orders, score customers, check policy, and find product info.")

# Initialize agent once per session
if "executor" not in st.session_state:
    with st.spinner("Loading agent..."):
        from agent.executor import build_executor
        st.session_state.executor = build_executor()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clean_output(text: str) -> str:
    """Remove undefined and stray markdown artifacts from agent output."""
    text = text.replace("undefined", "").strip()
    text = re.sub(r'```+', '', text).strip()
    return text

def display_score_result(output: str):
    """Check if output contains scoring data and display it cleanly."""
    try:
        json_match = re.search(r'\{.*?"predicted_ltv".*?\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())

            st.subheader("Customer Score")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted LTV", f"${data.get('predicted_ltv', 'N/A')}")
            with col2:
                tier = data.get('risk_tier', 'unknown')
                st.metric("Risk Tier", tier.upper())
            with col3:
                st.metric("Order ID", data.get('order_id', 'N/A'))

            if tier == "high":
                st.success("✅ High value customer — consider priority follow-up")
            elif tier == "medium":
                st.info("ℹ️ Medium value customer — standard handling")
            else:
                st.warning("⚠️ Low value customer — monitor for churn risk")

            if data.get('top_factors'):
                st.write("**Top factors:**")
                for factor in data.get('top_factors', []):
                    st.write(f"- {factor}")

            if data.get('suggested_next_step'):
                st.write(f"**Suggested next step:** {data.get('suggested_next_step')}")

            return True
    except Exception:
        pass

    # If no JSON found, check if it looks like a scoring response and clean it up
    if "predicted to spend" in output.lower() or "risk tier" in output.lower() or "ltv" in output.lower():
        # Clean up the text and display it properly
        clean = output.replace("$", "\\$")
        st.markdown(clean)
        return True

    return False

# Handle new input
if prompt := st.chat_input("Ask about an order, policy, or product..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.executor.invoke({"input": prompt})
                output = clean_output(response["output"])

                # Try to display as structured score card first
                displayed_as_card = display_score_result(output)

                # If not a score card just display as text
                if not displayed_as_card:
                    st.write(output)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": output
                })

            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
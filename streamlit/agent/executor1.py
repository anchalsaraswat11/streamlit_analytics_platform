from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', '.env'), override=True)

import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import hub

from agent.tools.lookup import lookup_order
from agent.tools.scoring import score_order
from agent.tools.policy_rag import query_policy
from agent.tools.product import lookup_product

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def build_executor():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=OPENAI_API_KEY
    )

    tools = [
        Tool(
            name="OrderLookup",
            func=lookup_order,
            description="Look up an order by order_id. Returns the raw feature row for that order. Use this when asked about a specific order."
        ),
        Tool(
            name="ScoreOrder",
            func=score_order,
            description="Score an order and predict customer LTV. Input is an order_id. Returns a JSON object with predicted_ltv, risk_tier, order_id, top_factors, and suggested_next_step. Return the JSON exactly as received without paraphrasing."
        ),
        Tool(
            name="PolicyRAG",
            func=query_policy,
            description="Answer questions about company policy. Input is a plain English question. Returns relevant policy text with citations. Never invent policy."
        ),
        Tool(
            name="ProductInfo",
            func=lookup_product,
            description="Look up product details by product_id. Returns category, subcategory, warranty, return window and other product info."
        )
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    system_prompt = """You are an internal business assistant for an e-commerce company.
You help support leads, risk analysts, and merchandising staff.
You can: look up orders, score customer LTV, answer policy questions, look up product info.
You cannot: process refunds, make account changes, or take any actions — inform only.
If asked something outside these capabilities, politely decline.
Always cite the policy section when answering policy questions.
Never fabricate a score — only return one if the SageMaker endpoint was called successfully."""

    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
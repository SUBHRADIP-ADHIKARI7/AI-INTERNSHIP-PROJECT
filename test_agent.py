import os
from dotenv import load_dotenv

import langchain
langchain.debug = True

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import search_internships
from app import SYSTEM_PROMPT

load_dotenv()

print("Setting up agent...")
llm = ChatOllama(model="mistral", temperature=0.4)
tools = [search_internships]
agent = create_react_agent(llm, tools)

history = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="Find me Python developer internships")
]

print("Invoking agent...")
try:
    result = agent.invoke({"messages": history})
    print("\n\nFINAL RESULT:")
    for m in result["messages"]:
        print(type(m).__name__, ":", m.content)
except Exception as e:
    print("\n\nERROR:", e)

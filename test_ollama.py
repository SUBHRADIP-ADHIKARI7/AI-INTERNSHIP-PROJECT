from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

print("Initializing ChatOllama...")
try:
    llm = ChatOllama(model="mistral", temperature=0.4)
    print("Invoking...")
    res = llm.invoke([HumanMessage(content="Hello!")])
    print("Response:", res.content)
except Exception as e:
    print("Error:", e)

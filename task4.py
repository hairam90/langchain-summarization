import os
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import numpy as np

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

sp = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 1 sentence:\n\n{text}"
)

chain = sp | llm

@tool
def text_summarizer(text: str) -> str:
    """Summarize the given text into a single sentence."""
    return chain.invoke({"text": text}).content

'''agent = initialize_agent(
    tools=[text_summarizer],
    llm=llm,  
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
'''
agent_node = create_react_agent(llm, [text_summarizer])
# agent = AgentExecutor(agent_node, verbose=True)

healthcare_text = """
Artificial Intelligence has significantly impacted healthcare by enabling faster and more accurate diagnoses through
imaging analysis, predicting patient deterioration, optimizing hospital workflows, and personalizing treatments 
using data-driven insights. It also plays a crucial role in accelerating drug discovery and improving patient 
monitoring through wearable devices and AI assistants.
"""

print("\n--- Test 1: Impact of AI on Healthcare ---\n")
# response1 = agent.invoke(f"Summarize the following text: {healthcare_text}")
# print(response1)
response = agent_node.invoke(
    {"messages": [HumanMessage(content=f"Summarize the following text: {healthcare_text}")]}
)
print(response['messages'][-1].content)

print("\n--- Test 2: Summarize something interesting ---\n")
response = agent_node.invoke(
    {"messages": [HumanMessage(content=f"Summarize something interesting")]}
)
print(response['messages'][-1].content)


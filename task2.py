import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

# 3-sentence summarization
sp3 = PromptTemplate(input_variables=["text"], template="Summarize the following text into exactly 3 sentences:\n\n{text}")

# 1-sentence summarization
sp1 = PromptTemplate(input_variables=["text"], template="Summarize the following text into exactly 1 sentence:\n\n{text}")

input_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science that aims to create 
machines capable of performing tasks that typically require human intelligence. 
These tasks include understanding natural language, recognizing patterns, learning from experience, 
solving problems, and even exhibiting creativity. In recent years, 
AI has seen widespread adoption across various industries such as healthcare, finance, transportation, 
and education, transforming the way businesses operate and people live. For example, in healthcare, 
AI-powered systems can analyze medical images and assist in diagnosing diseases with remarkable accuracy. 
In finance, algorithms can predict stock market trends and detect fraudulent transactions in real time. 
Autonomous vehicles, driven by AI, are set to revolutionize transportation by reducing accidents and 
improving efficiency. Moreover, AI is increasingly being integrated into everyday applications 
like virtual assistants, recommendation systems, and customer service chatbots. Despite its benefits, 
AI also raises concerns regarding ethics, privacy, and potential job displacement, which researchers and 
policymakers are actively addressing. As AI continues to advance, it is essential to ensure that its development 
remains aligned with human values and societal needs to foster a future where AI serves as a beneficial tool for all.
"""

print("\n--- 3 Sentence Summary ---\n")
chain3 = (sp3 | llm).invoke({"text": input_text}).content
print(chain3)

print("\n--- 1 Sentence Summary ---\n")
chain1 = (sp1 | llm).invoke({"text": input_text}).content
print(chain1)

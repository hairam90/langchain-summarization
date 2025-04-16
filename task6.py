from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

# 1-sentence summarization
sp1 = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    The previous summary is: {history}
    Based on that, summarize the following text in 1 sentence considering the context:\n\n{input}
    """
)

ml = """
Machine learning is a branch of artificial intelligence that enables computers to learn from data without being 
explicitly programmed. It uses algorithms to identify patterns, make decisions, and improve performance over time. 
Common types include supervised, unsupervised, and reinforcement learning. Applications of machine learning span 
various industries, including healthcare, finance, marketing, and autonomous systems. For example, it powers 
recommendation engines, fraud detection systems, and speech recognition tools. As more data becomes available, 
machine learning continues to evolve, offering new opportunities and challenges. Ensuring fairness, transparency, 
and ethical use remains critical as machine learning becomes increasingly integrated into daily life.
"""

dl="""
Deep learning is a subset of machine learning that focuses on neural networks with many layers, 
enabling computers to learn complex patterns and representations from large amounts of data. 
Inspired by the human brain, deep learning models excel at tasks like image recognition, 
natural language processing, and speech translation. They power applications such as self-driving cars, 
facial recognition, and virtual assistants. Deep learning requires substantial computational resources and 
large datasets but offers remarkable accuracy and performance. As technology advances, deep learning continues 
to push the boundaries of artificial intelligence, making it a driving force behind many cutting-edge innovations 
in today's digital world.
"""

# conversation buffer memory
memory=ConversationBufferWindowMemory(k=3, return_messages=True)
convo = ConversationChain(llm=llm, memory=memory, prompt=sp1, verbose=False)

print("\nSummarization with conversation buffer memory")

print("\n--- 1 Sentence Summary of machine learning text---\n")
print(convo.predict(input=ml))

print("\n--- 1 Sentence Summary of deep learning text---\n")
print(convo.predict(input=dl))

# conversation summary memory
memory2=ConversationSummaryMemory(llm=llm,return_messages=True)
convo2=ConversationChain(llm=llm, memory=memory2, prompt=sp1, verbose=False)

print("\nSummarization with conversation summary memory")

print("\n--- 1 Sentence Summary of machine learning text---\n")
print(convo2.predict(input=ml))

print("\n--- 1 Sentence Summary of deep learning text---\n")
print(convo2.predict(input=dl))
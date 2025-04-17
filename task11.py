import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from datetime import datetime

load_dotenv()

loader = TextLoader("ai_intro.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='.')
chunks = splitter.split_documents(docs)

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
)

vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 1 sentence:\n\n{text}"
)

summary_chain = summary_prompt | llm

# Tool 1: Retriever Tool
@tool
def retrieve_text(query: str) -> str:
    """Retrieve the most relevant text from the document based on the query."""
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

# Tool 2: Summarizer Tool
@tool
def summarize_text(text: str) -> str:
    """Summarize the given text into a single sentence."""
    return summary_chain.invoke({"text": text}).content

# Tool 3: Word Count Tool
@tool
def count_words(text: str) -> str:
    """Count the number of words in the provided text."""
    word_count = len(text.split())
    return f"Word count: {word_count}"

# Tool 4: Date Time tool
@tool
def tdatetime(text: str) -> str:
    """give today's date"""
    today = datetime.today().date()
    return f"Today's date is: {today}"

@tool
def websearch(text: str) -> str:
    """give we search results"""
    res = websearchfunc()
    return res

def websearchfunc():
    return f"Artificial Intelligence (AI) is the simulation of human intelligence by machines, enabling them to learn, reason, and make decisions. AI powers technologies like chatbots, self-driving cars, and recommendation systems. It continues to revolutionize industries by improving efficiency, enhancing user experiences, and unlocking new possibilities across science, healthcare, finance, and more."

# Create Agent
tools = [retrieve_text, summarize_text, count_words, tdatetime, websearch]
agent = create_react_agent(llm, tools)

response = agent.invoke(
    {"messages": [HumanMessage(content=f"Summarize the text about AI and tell me today's date: {loader}")]}
)
print(response['messages'][-1].content)

response2 = agent.invoke(
    {"messages": [HumanMessage(content=f"Summarize AI trends and search for recent updates.: {loader}")]}
)
print(response2['messages'][-1].content)

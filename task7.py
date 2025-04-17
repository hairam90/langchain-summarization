import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

load_dotenv()
splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=30, separator='.')

embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
)

prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 1 sentence:\n\n{text}"
)

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

# for PDF
#load pdf data
pdf_loader = PyPDFLoader("C:\\Users\\imperium dynamics\Desktop\\langchain summarization\AI Ethics.pdf")
pdf_data = pdf_loader.load()

# split into chunks
pdf_chunks = splitter.split_documents(pdf_data)

# vector stores
pdf_vectorstore = Chroma.from_documents(pdf_data, embedding=embeddings)
pdf_retriever = pdf_vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})
pdf_retrieved_docs = pdf_retriever.invoke("AI challenges")
pdf_retrieved_text = "\n".join([doc.page_content for doc in pdf_retrieved_docs])

# summary
pdf_summary = (prompt | llm).invoke({"text": pdf_retrieved_text}).content

print("\n--- PDF Summary ---\n")
print(pdf_summary)

# for WEB
web_loader = WebBaseLoader(r"https://www.coursera.org/articles/ai-trends")
web_data = web_loader.load()

# split into chunks
web_chunks = splitter.split_documents(web_data)

# vector stores
web_vectorstore = Chroma.from_documents(web_data, embedding=embeddings)
web_retriever = web_vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})
web_retrieved_docs = web_retriever.invoke("AI challenges")
web_retrieved_text = "\n".join([doc.page_content for doc in web_retrieved_docs])

# summary
web_summary = (prompt | llm).invoke({"text": web_retrieved_text}).content

print("\n--- Web Summary ---\n")
print(web_summary)

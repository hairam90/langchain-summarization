import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

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

vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 1})
retrieved_docs = retriever.invoke("AI milestones")
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n\n{text}"
)

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

summary = (prompt | llm).invoke({"text": retrieved_text}).content
print("\n--- Summary ---\n")
print(summary)

sp1 = PromptTemplate(input_variables=summary, template="What's the key event mentioned in following text:\n\n{summary}")
out1 = (prompt | llm).invoke({"text": summary}).content
print("\n--- Key event finding in summary ---\n")
print(out1)

sp2 = PromptTemplate(input_variables=["text"], template="What's the key event mentioned in the following text:\n\n{text}")
out2 = (prompt | llm).invoke({"text": retrieved_text}).content
print("\n--- key event finding in whole text ---\n")
print(out2)



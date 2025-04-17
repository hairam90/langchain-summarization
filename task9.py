import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Load and split the document
loader = TextLoader("ai_intro.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='.')
chunks = splitter.split_documents(docs)

# Setup embeddings and LLM
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
)

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

# Create the vector store
vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

# Generate rephrased queries
rephrase_prompt = PromptTemplate(
    input_variables=["question"],
    template="Generate 3 diverse rephrasings of the following question:\n\n{question}"
)

rephrased_output = llm.invoke(rephrase_prompt.format(question="AI advancements"))
alt_queries = [q.strip('- ').strip() for q in rephrased_output.content.strip().split("\n") if q.strip()]
all_queries = ["AI advancements"] + alt_queries

print("\n--- All Queries ---\n")
for i, q in enumerate(all_queries, 1):
    print(f"{i}. {q}")

# Retrieve and print results for each query
retrieved_docs = []
seen_texts = set()

print("\n--- Retrieved Text per Query ---\n")

for i, q in enumerate(all_queries, 1):
    results = vectorstore.similarity_search(q, k=5)
    print(f"Query {i}: {q}")
    for doc in results:
        if doc.page_content not in seen_texts:
            seen_texts.add(doc.page_content)
            retrieved_docs.append(doc)
            print(doc.page_content)
        else:
            pass
    print()

# Combine and print final result
combined_text = "\n".join(doc.page_content for doc in retrieved_docs)

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 1 sentence:\n\n{text}"
)

summary = (summary_prompt | llm).invoke({"text": combined_text}).content

print("\n--- Combined Retrieved Text ---\n")
print(combined_text)

print("\n--- Summary of All Retrieved Content ---\n")
print(summary)

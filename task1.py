import langchain
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("API_VERSION")

print("AZURE_OPENAI_API_KEY:", api_key)
print("ENDPOINT_URL:", endpoint)
print("DEPLOYMENT_NAME:", deployment)
print("API_VERSION:", api_version)
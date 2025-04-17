import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.5,
)

response_schemas = [
    ResponseSchema(name="summary", description="A concise summary of the input text."),
    ResponseSchema(name="length", description="Character count of the summary."),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following text into exactly 1 sentence. Then return a JSON object with:
- 'summary': the one-sentence summary
- 'length': number of characters in the summary
{format_instructions}
Text to summarize:
{text}
""",
    partial_variables={"format_instructions": format_instructions}
)

input_text = """
Artificial Intelligence (AI) is being increasingly applied across various sectors. In healthcare, AI helps in early disease detection,
drug discovery, and robotic-assisted surgeries. In finance, AI powers fraud detection systems, risk assessments, and automated trading.
Retail businesses use AI for inventory optimization, personalized marketing, and customer sentiment analysis. AI enhances education 
through personalized learning platforms, automated grading, and intelligent tutoring systems. Autonomous vehicles rely on AI for object 
detection and real-time decision-making. Agriculture benefits from AI through crop monitoring, yield prediction, and precision farming.
In cybersecurity, AI is used to detect anomalies and respond to threats in real time. Smart assistants like Alexa and Siri rely on 
natural language processing and machine learning to interact with users. AI is also crucial in environmental science for climate 
modeling and tracking deforestation. As AI continues to evolve, its integration into everyday life is expected to increase, improving 
efficiency, productivity, and decision-making.
"""

chain = prompt | llm | parser
output = chain.invoke({"text": input_text})

print("\n--- Structured Summary Output ---\n")
print(output)

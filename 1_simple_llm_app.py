from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

llm = OpenAI(model = "gpt-3.5-turbo-instruct" ,temperature=0.9)

# create a prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy title for {topic}",
)

# define an input
topic = input("Enter a topic: ")

# generate a response
formatted_prompt = prompt.format(topic=topic)

# generate a response directly from llm
response = llm.invoke(formatted_prompt)

print("Generated title for ", topic," is ", response)
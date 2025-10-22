from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(topic)

print("Generated title for ", topic," is ", response)
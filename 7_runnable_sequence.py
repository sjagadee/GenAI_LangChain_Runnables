from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt_1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Explain about the given joke: {joke}",
    input_variables=['joke']
)

model = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

print(chain.invoke({'topic': 'AI'}))
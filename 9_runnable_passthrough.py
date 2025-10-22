from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel
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

joke_gen_chain = RunnableSequence(prompt_1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "joke_explaination": RunnableSequence(prompt_2, model, parser)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(chain.invoke({'topic': 'AI'}))
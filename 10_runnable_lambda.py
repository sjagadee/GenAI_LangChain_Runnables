from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt_1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

model = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()

def word_counter(text):
    return len(text.split())

runnable_wordcount = RunnableLambda(word_counter)

joke_gen_chain = RunnableSequence(prompt_1, model, parser)

# parallel_chain = RunnableParallel({
#     "joke": RunnablePassthrough(),
#     "joke_word_count": runnable_wordcount
# })

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "joke_word_count": RunnableLambda(lambda x: len(x.split()))
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({'topic': 'Machine Learning'})

print(result)
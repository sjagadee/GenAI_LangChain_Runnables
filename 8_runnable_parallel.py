from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt_1 = PromptTemplate(
    template="Generate a tweet about this topic: {topic}",
    input_variables=["topic"]
)

prompt_2 = PromptTemplate(
    template="Generate an LinkedIn post about this topic: {topic}",
    input_variables=["topic"]
)

model_gpt = ChatOpenAI(model="gpt-3.5-turbo")
model_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

parser = StrOutputParser()

parallel_chains = RunnableParallel({
    'tweet': RunnableSequence(prompt_1, model_gpt, parser),
    'linkedin_post': RunnableSequence(prompt_2, model_gemini, parser)
})

result = parallel_chains.invoke({'topic': 'AI'})
print(result)
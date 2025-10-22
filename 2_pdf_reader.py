from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI

# load a document
loader = TextLoader('docs.txt')
documents = loader.load()

# split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# convert text to vector embeddings and store in FAISS vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# create a retriever
retriever = vectorstore.as_retriever()

# Manually retrieve releavant documents
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

# combine retrieved documents into a single string
retrieved_docs = "\n".join([doc.page_content for doc in retrieved_docs])

# generate response
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

# generate a response
prompt = f'Based on the following document, {retrieved_docs}, answer the following question: {query}'
response = llm.invoke(prompt)

print(response)
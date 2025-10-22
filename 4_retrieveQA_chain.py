from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

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

# # generate response
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

# create a qa chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

query = "What are the key takeaways from the document?"
response = qa_chain.run(query)

print(response)
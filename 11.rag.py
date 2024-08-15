# Langchain provides Document Transformers that allow you to easily
#   split strings from Document page_content into chunks.

# Langchain supports many text embeddings,
#   that can directly convert string text to an embedded vectorized representation.

# Different embedding models can not interact with each other,
#   meaning you would need to re-embed an entire set of documents
#   if you were to switch embedding models in the future

# Vector Store Integrations:
# Just like with LLMs and Chat Models, Langchain offers many different
#   options for vector stores!
# We will use an open-source and free vector store called Chroma,
#   which has great integrations with Langchain.

# As we begin to link Langchain objects,
#   there are times when we need to pass in Vector Stores as retriever objects,
#   which can be easily done via a as_retriever() method call.

import os

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(current_dir, "db")


output_parser = StrOutputParser()
model = ChatOllama(model="mistral")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("------------------------------------")
print("Fetching data from WikipediaLoader:-")
print("------------------------------------")
documents = WikipediaLoader(query="Ratan Tata", load_max_docs=2).load()
print(documents)


print("--------------------------------------------")
print("Splitting data using CharacterTextSplitter:-")
print("--------------------------------------------")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)
docs = text_splitter.split_documents(documents=documents)
print(docs)


print("-----------------------------------------")
print("Embedding docs and storing in Chroma db:-")
print("-----------------------------------------")
db_connection = Chroma.from_documents(documents=docs,
                                      embedding=embeddings,
                                      persist_directory=persist_directory)
print("Embedding Successfull!!!")


print("--------------------------------")
print("Retrieving docs from Chroma db:-")
print("--------------------------------")
retriever = db_connection.as_retriever()
response = retriever.invoke("When was Ratan Tata born?")
print(response)


print("-------------------------------------")
print("Getting answer from db using Model :-")
print("-------------------------------------")
prompt_template = PromptTemplate(
    input_variables=["document"],
    template="Answer any question from this document: {document}. When was Ratan Tata born?")
chain = prompt_template | model | output_parser
result = chain.invoke({"document": response[0].page_content})
print(result)

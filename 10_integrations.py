from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
model = ChatOllama(model="mistral")
docs = WikipediaLoader(query="Indian Independence").load()

# print(docs[0].page_content)

human_prompt = HumanMessagePromptTemplate.from_template(
    'Please give me a single sentence summary of the following:\n{document}')
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])


print("--------------------------------")
print("Chatmodel using WikipediaData :-")
print("--------------------------------")
chain = chat_prompt | model | output_parser
result = chain.invoke({"document": docs[0].page_content})
print(result)

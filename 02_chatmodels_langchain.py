# Chat Models have a series of messages, just like a chat text thread,
#   except one side of the conversation is an AI LLM.
# Langchain creates 3 schema objects for this:
#   SystemMessage - General system tone or personality
#   HumanMessage - Human request or reply
#   AIMessage - AI’s reply

from langchain_ollama import ChatOllama
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

chat = ChatOllama(model="dolphin-mistral")

print("------------------------------")
print("ChatModel with Human Message:-")
print("------------------------------")
result = chat.invoke(
    [HumanMessage(content="Can you tell me a fact about Tamil?")])
# print(result)
print(result.content)


print("-------------------------------------------------")
print("ChatModel with Human Message and System Message:-")
print("-------------------------------------------------")
result = chat.invoke([SystemMessage(content="You are a very rude South Indian Women who doesn't want to answer questions"),
                      HumanMessage(content="Can you tell me a fact about Tamil?")])
print(result.content)


print("----------------------------------------------------------------")
print("ChatModel with Human Message and System Message using generate:-")
print("----------------------------------------------------------------")
result = chat.generate(
    [
        [SystemMessage(content="You are a polite South Indian Women"),
         HumanMessage(content="Can you tell me a fact about Tamil?")]
    ]
)
print(result.generations[0][0].text)

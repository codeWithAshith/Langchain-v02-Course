from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOllama(model="mistral")


print("---------------------------------")
print("Chatmodel using Prompt Template:-")
print("---------------------------------")
system_template = "You are an AI recipe assistant that specializes in {dietary_preference} dishes that can be prepared in {cooking_time}."
system_message_prompt = SystemMessagePromptTemplate.from_template(
    system_template)

human_template = "{recipe_request}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

prompt = chat_prompt.format_prompt(cooking_time="15 min",
                                   dietary_preference="Vegan", recipe_request="Quick Snack").to_messages()

result = chat.invoke(prompt)
print(result.content)

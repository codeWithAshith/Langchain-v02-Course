# The output of the previous runnable's .invoke() call is passed as input
#   to the next runnable.
# This can be done using the pipe operator (|), or the more explicit

from langchain_core.prompts import (PromptTemplate,
                                    ChatPromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    )
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="mistral")
template = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fact about {topic}"
)


print("-------------------------------------")
print("Prompt Template without using chain:-")
print("-------------------------------------")
prompt = template.format(topic='Tamil')
result = model.invoke(prompt)
print(result)


print("------------------------------")
print("Prompt Template using chain:-")
print("------------------------------")
chain = template | model
result = chain.invoke({"topic": "Tamil"})
print(result)


chat = ChatOllama(model="mistral")
system_template = "You are an AI recipe assistant that specializes in {dietary_preference} dishes that can be prepared in {cooking_time}."
system_message_prompt = SystemMessagePromptTemplate.from_template(
    system_template)

human_template = "{recipe_request}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])


print("-----------------------------------------------------")
print("Chatmodel using Prompt Template without using chain:-")
print("-----------------------------------------------------")
prompt = chat_prompt.format_prompt(cooking_time="15 min",
                                   dietary_preference="Vegan", recipe_request="Quick Snack").to_messages()
result = chat.invoke(prompt)
print(result.content)


print("-------------------------------------------------------------")
print("Chatmodel using Prompt Template using chain & output parser:-")
print("-------------------------------------------------------------")
chain = chat_prompt | model | StrOutputParser()
result = chain.invoke({"cooking_time": "15 min",
                      "dietary_preference": "Vegan", "recipe_request": "Quick Snack"})
print(result)

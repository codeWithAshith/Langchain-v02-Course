from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="mistral")


print("------------------------------------")
print("LLM using Prompt Template No input:-")
print("------------------------------------")
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a fact")
prompt = no_input_prompt.format()
result = llm.invoke(prompt)
print(result)


print("-------------------------------------")
print("LLM using Prompt Template one input:-")
print("-------------------------------------")
one_input_prompt = PromptTemplate(
    input_variables=["topic"], template="Tell me a fact about {topic}.")
prompt = one_input_prompt.format(topic="Tamil")
result = llm.invoke(prompt)
print(result)


print("------------------------------------------")
print("LLM using Prompt Template multiple input:-")
print("------------------------------------------")
multiple_input_prompt = PromptTemplate(
    input_variables=["topic", "level"],
    template="Tell me a fact about {topic} for a student {level} level."
)
prompt = multiple_input_prompt.format(
    topic='Tamil', level='1th standard')
result = llm.invoke(prompt)
print(result)

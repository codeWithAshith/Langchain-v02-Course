# LangChain is a framework for developing applications
#   that are powered by Large Language Models,
#   such as OpenAI’s GPT models or Google’s PaLM-2 or Gemini LLMs.
# The framework does this through the use of Modules
#   (also sometimes referred to as Components).

# At its core Langchain needs to be able to send text to LLMs and
#   also receive and work with their outputs.

# Using Langchain for Model IO will later allow us to build chains,
#   but also give us more flexibility in switching LLM providers in the future,
#   since the syntax is standardized across LLMs and only the parameters or arguments provided change.


# There are two main types of APIs in Langchain:
#   1. LLM - Text Completion Model: Returns the most likely text to continue
#   2. Chat - Converses with back and forth messages, can also have a “system” prompt.


from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="mistral")

print("------------------")
print("LLM using invoke:-")
print("------------------")
result = llm.invoke("Give me a fact about tamil:")
print(result)

print("---------------------")
print("LLM using generate:-")
print("---------------------")
result = llm.generate(["Give me a fact about tamil:",
                       "Give me a fact about devops:"]
                      )
# print(result.generations)
print(result.generations[1][0].text)

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

model = OllamaLLM(model="mistral")

print("--------------------------------")
print("CommaSeparatedListOutputParser:-")
print("--------------------------------")
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | model | output_parser
result = chain.invoke({"subject": "ice cream flavors"})
print(result)


print("----------------------")
print("JsonOutputParser:-")
print("----------------------")


class Product(BaseModel):
    name: str = Field(description="name of the product")
    description: str = Field(description="description of the product")
    price: float = Field(description="price of the product")


output_parser = JsonOutputParser(pydantic_object=Product)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},

)

chain = prompt | model | output_parser
result = chain.invoke({"query": "Give me a dummy product object"})
print(result)

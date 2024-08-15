# Sometimes it’s easier to give the LLM a few examples of input/output pairs
#   before sending your main request.
# This allows the LLM to “learn” the pattern you are looking for
#   and may lead to better results.
# It should be noted that there is currently no consensus on best practices,
#   but LangChain recommends building a history of Human and AI message inputs.

from langchain_ollama import ChatOllama
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOllama(model="mistral")


print("---------------------------------")
print("Few Shots using Prompt Template:-")
print("---------------------------------")
template = "You simplify complex cooking instructions."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

complex_instruction = "Preheat the oven to 375°F (190°C). In a mixing bowl, combine flour, sugar, baking powder, and salt. In another bowl, beat eggs, then mix in milk and melted butter. Gradually add wet ingredients to dry ingredients, stirring until just combined. Pour batter into a greased baking dish and bake for 25-30 minutes or until golden brown."
example_input_one = HumanMessagePromptTemplate.from_template(
    complex_instruction)

simple_instruction = "Preheat oven to 375°F. Mix flour, sugar, baking powder, and salt. In another bowl, beat eggs, then add milk and melted butter. Combine wet and dry ingredients. Pour into a greased dish and bake for 25-30 minutes."
example_output_one = AIMessagePromptTemplate.from_template(simple_instruction)

human_template = "{recipe_text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_input_one,
        example_output_one, human_message_prompt]
)

some_example_text = "Heat the oil in a large pan over medium heat. Add chopped onions and cook until translucent. Add minced garlic and cook for another minute. Stir in diced tomatoes and cook until they break down. Add ground beef and cook until browned. Season with salt, pepper, and herbs. Simmer for 20 minutes."
request = chat_prompt.format_prompt(
    recipe_text=some_example_text).to_messages()

result = chat.invoke(request)

print(result.content)

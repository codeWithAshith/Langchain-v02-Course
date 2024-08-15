# Langchain comes with built-in loader tools
#   to quickly load in files to its own Document object.
# Note that many of these loaders require other libraries,
#   for example PDF loading requires the pypdf library and
#   HTML loading requires the Beautiful Soup library.

# pip install -U langchain-community

from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("asset/Sample.csv", encoding="UTF-8")
data = loader.load()

# print(data)
print(data[0].page_content)

# !pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("asset/Sample.pdf")
pages = loader.load_and_split()

print(pages)
# print(pages[0].page_content)

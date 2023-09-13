import os
import sys

# from langchain.llms import OpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, JSONLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator

# from langchain.chains import ConversationalRetrievalChain


# load all environmental variables
load_dotenv()
# load api key into app
os.getenv("OPENAI_API_KEY")

# llm = OpenAI()
chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# TextLoader
# loader_kwargs={"jq_schema": ".content", "json_lines": True}
# loader = DirectoryLoader(
#     "data/",
#     loader_cls=JSONLoader,
#     glob="**/*.json",
#     loader_kwargs={
#         "jq_schema": ".[]",
#     },
# )
loader2 = DirectoryLoader("data/", loader_cls=TextLoader)

# docs = loader.load()
docs = loader2.load()
# print(f"docs count {len(docs)}")
# print(f"docs: {docs[0]}")


# index = VectorstoreIndexCreator().from_loaders([loader])
index = VectorstoreIndexCreator().from_loaders([loader2])
query = input("Prompt: ")
result = index.query(query)
print(result)

# result = chat_model.predict(query)
# print(result)

# chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(model="gpt-3.5-turbo"),
#     retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
# )
# chat_history = []
# while True:
#     if not query:
#         query = input("Prompt: ")
#     if query in ["quit", "q", "exit"]:
#         sys.exit()
#     result = chain({"question": query, "chat_history": chat_history})
#     print(result["answer"])

#     chat_history.append((query, result["answer"]))
#     query = None

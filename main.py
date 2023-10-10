import os
import sys

# from langchain.llms import OpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, JSONLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# from langchain.chains import ConversationalRetrievalChain


# load all environmental variables
load_dotenv()
# load api key into app
os.getenv("OPENAI_API_KEY")

# llm = OpenAI()
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


loader = DirectoryLoader(
    "data/",
    loader_cls=JSONLoader,
    glob="**/*.json",
    loader_kwargs={
        "jq_schema": ".[].name",
    },
)
# loader2 = DirectoryLoader("data/", loader_cls=TextLoader)

docs = loader.load()
# docs = loader2.load()
# print(f"docs count {len(docs)}")
print(f"docs: {docs}")


if os.path.exists("persist"):
    print("os path")
    vectorstore = Chroma(
        persist_directory="persist", embedding_function=OpenAIEmbeddings()
    )
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    print("else")
    index = VectorstoreIndexCreator(
        # vectorstore_cls=Chroma,
        # embedding=OpenAIEmbeddings(),
        vectorstore_kwargs={"persist_directory": "persist"},
    ).from_loaders([loader])
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

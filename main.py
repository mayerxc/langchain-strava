import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import utils as chroma_utils

# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.llms import OpenAI
# from langchain.chains import ConversationalRetrievalChain


# load all environmental variables
load_dotenv()
# load api key into app
os.getenv("OPENAI_API_KEY")

# llm = OpenAI()
# temperature=0.5
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")


# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["distance_in_meters"] = record.get("distance")
    metadata["start_date_local"] = record.get("start_date_local")
    metadata["moving_time_in_seconds"] = record.get("moving_time")
    metadata["location_city"] = record.get("location_city")
    metadata["location_state"] = record.get("location_state")
    metadata["location_country"] = record.get("location_country")
    metadata["sport_type"] = record.get("sport_type")
    metadata["average_heartrate_in_beats_per_minute"] = record.get("average_heartrate")
    metadata["max_heartrate"] = record.get("max_heartrate")
    metadata["total_elevation_gain_in_meters"] = record.get("total_elevation_gain")
    return metadata


json_loader_kwargs = {
    "jq_schema": ".[]",
    "content_key": "name",
    "metadata_func": metadata_func,
}
loader = DirectoryLoader(
    "data/",
    loader_cls=JSONLoader,
    glob="**/*.json",
    loader_kwargs=json_loader_kwargs,
)
# loader2 = DirectoryLoader("data/", loader_cls=TextLoader)

docs = loader.load()
# filter docs because of unexpected metadata error for ChromaDB
filtered_metadata_docs = chroma_utils.filter_complex_metadata(docs)
# docs = loader2.load()
# print(f"docs count {len(docs)}")

print(f"docs: {docs}")


if os.path.exists("persist"):
    print("os path")
    vectorstore = Chroma(
        persist_directory="persist", embedding_function=OpenAIEmbeddings()
    )
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    # retriever = vectorstore.as_retriever()
else:
    print("else")
    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=OpenAIEmbeddings(),
        vectorstore_kwargs={"persist_directory": "persist"},
    ).from_documents(filtered_metadata_docs)
query = input("Prompt: ")
result = index.query_with_sources(query)
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

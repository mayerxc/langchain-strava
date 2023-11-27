import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import utils as chroma_utils
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


# load all environmental variables
load_dotenv()
# load api key into app
os.getenv("OPENAI_API_KEY")

# gpt-4
# gpt-3.5-turbo
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")


# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["distance_in_meters"] = record.get("distance")
    metadata["average_speed"] = record.get("average_speed")
    metadata["start_date_local"] = record.get("start_date_local")
    metadata["moving_time_in_seconds"] = record.get("moving_time")
    metadata["location_city"] = record.get("location_city")
    metadata["location_state"] = record.get("location_state")
    metadata["location_country"] = record.get("location_country")
    metadata["sport_type"] = record.get("sport_type")
    metadata["average_heartrate"] = record.get("average_heartrate")
    metadata["max_heartrate"] = record.get("max_heartrate")
    metadata["total_elevation_gain_in_meters"] = record.get("total_elevation_gain")
    return metadata


metadata_field_info = [
    AttributeInfo(
        name="distance_in_meters",
        description="The distance of the run or workout in meters",
        type="float",
    ),
    AttributeInfo(
        name="average_speed",
        description="The average speed of the run or workout in meters per second",
        type="float",
    ),
    AttributeInfo(
        name="start_date_local",
        description="Time of day that the workout started",
        type="date",
    ),
    AttributeInfo(
        name="moving_time_in_seconds",
        description="The duration of the workout",
        type="float",
    ),
    AttributeInfo(
        name="location_city",
        description="The city where the workout occurred",
        type="string",
    ),
    AttributeInfo(
        name="location_state",
        description="The state inside a country where the workout occurred",
        type="string",
    ),
    AttributeInfo(
        name="location_country",
        description="The country where the workout occurred",
        type="string",
    ),
    AttributeInfo(
        name="sport_type",
        description="The type of sport that the GPS watch recorded",
        type="string",
    ),
    AttributeInfo(
        name="average_heartrate",
        description="Average heart rate in beats per minute",
        type="float",
    ),
    AttributeInfo(
        name="max_heartrate",
        description="Maximum heart rate in beats per minute that occurred during the workout",
        type="float",
    ),
    AttributeInfo(
        name="total_elevation_gain_in_meters",
        description="The elevation gain of the run or workout",
        type="float",
    ),
]

json_loader_kwargs = {
    "jq_schema": ".[]",
    "content_key": "name",
    "metadata_func": metadata_func,
}
loader = DirectoryLoader(
    "data2/",
    loader_cls=JSONLoader,
    glob="**/*.json",
    loader_kwargs=json_loader_kwargs,
)

docs = loader.load()
# filter docs because of unexpected metadata error for ChromaDB
filtered_metadata_docs = chroma_utils.filter_complex_metadata(docs)

print(f"docs: {docs[0]}")


vectorstore = Chroma(
    persist_directory="persist", embedding_function=OpenAIEmbeddings()
).from_documents(filtered_metadata_docs, OpenAIEmbeddings())


retriever = SelfQueryRetriever.from_llm(
    llm=chat_model,
    vectorstore=vectorstore,
    document_contents="Workout data from a GPS watch",
    metadata_field_info=metadata_field_info,
)
prompt = input("Prompt: ")

found_docs = retriever.invoke(prompt)

print(found_docs)

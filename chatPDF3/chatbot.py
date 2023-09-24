import os
from softtek_llm.chatbot import Chatbot
from softtek_llm.models import OpenAI
from softtek_llm.cache import Cache
from softtek_llm.vectorStores import PineconeVectorStore
from softtek_llm.embeddings import OpenAIEmbeddings
from softtek_llm.schemas import Filter
from dotenv import load_dotenv

class OpenAIChatbot:
    def __init__(self):
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
        if OPENAI_API_BASE is None:
            raise ValueError("OPENAI_API_BASE not found in .env file")

        OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
        if OPENAI_EMBEDDINGS_MODEL_NAME is None:
            raise ValueError("OPENAI_EMBEDDINGS_MODEL_NAME not found in .env file")

        OPENAI_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME")
        if OPENAI_CHAT_MODEL_NAME is None:
            raise ValueError("OPENAI_CHAT_MODEL_NAME not found in .env file")
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if PINECONE_API_KEY is None:
            raise ValueError("PINECONE_API_KEY not found in .env file")

        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        if PINECONE_ENVIRONMENT is None:
            raise ValueError("PINECONE_ENVIRONMENT not found in .env file")

        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        if PINECONE_INDEX_NAME is None:
            raise ValueError("PINECONE_INDEX_NAME not found in .env file")

        vector_store = PineconeVectorStore(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE_INDEX_NAME,
        )
        embeddings_model = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDINGS_MODEL_NAME,
            api_type="azure",
            api_base=OPENAI_API_BASE,
        )
        cache = Cache(
            vector_store=vector_store,
            embeddings_model=embeddings_model,
        )
        model = OpenAI(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_CHAT_MODEL_NAME,
            api_type="azure",
            api_base=OPENAI_API_BASE,
            verbose=True,
        )
        """
        filters = [
            Filter(
                type="DENY",
                case="ANYTHING about the Titanic. YOU CANNOT talk about the Titanic AT ALL.",
            )
        ]"""
        self.chatbot = Chatbot(
            model=model,
            description="You are a very helpful and polite chatbot",
            #filters=filters,
            cache=cache,
            verbose=True,
        )
        
    
    def sendMessage(self, inputSring):
        response = self.chatbot.chat(
            inputSring,
            #print_cache_score=True,
            #cache_kwargs={"namespace": "chatbot-test"},
        )
        print("Chatbot:")
        print(response.message.content)

chatbot1 = OpenAIChatbot()
print("Mensaje:")
inputStr = input()
chatbot1.sendMessage(inputStr)
del chatbot1
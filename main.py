import os
from dotenv import load_dotenv

from fastapi import FastAPI

load_dotenv()

vec_db_connection_string = os.getenv("VEC_DB_CONNETION_STR")
db_name = os.getenv("DB_NAME")
doc_collection_name = os.getenv("DOC_COLLECTION_NAME")
startups_collection_name = os.getenv("STARTUP_COLLECTION_NAME")
vector_index_name = os.getenv("VECTOR_INDEX_NAME")

azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_openai_embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBDNG_DEPLOYMENT_NAME")

azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_llm_endpoint = os.getenv("AZURE_OPENAI_LLM_ENDPOINT")
azure_openai_embdng_endpoint = os.getenv("AZURE_OPENAI_EMBDNG_ENDPOINT")
azure_openai_api_type = os.getenv("AZURE_OPENAI_API_TYPE")
azure_openai_llm_api_key = os.getenv("AZURE_OPENAI_LLM_API_KEY")
azure_openai_embedding_api_key = os.getenv("AZURE_OPENAI_EMBDNG_API_KEY")

app = FastAPI()

import urls

app.include_router(urls.router, prefix="/korean-startups", tags=['Korean Startups'])
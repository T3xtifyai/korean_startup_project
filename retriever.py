from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_mongodb import MongoDBAtlasVectorSearch

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from main import (
    vec_db_connection_string, db_name, doc_collection_name, startups_collection_name, vector_index_name,
    azure_openai_deployment_name, azure_openai_embedding_deployment_name,
    azure_openai_api_version, azure_openai_llm_endpoint, azure_openai_embdng_endpoint,
    azure_openai_api_type, azure_openai_llm_api_key, azure_openai_embedding_api_key
)

import prompts

embedding_model = AzureOpenAIEmbeddings(
        disallowed_special=(), model="text-embedding-3-small", dimensions=512,
        api_key=azure_openai_embedding_api_key,
        api_version=azure_openai_api_version,
        openai_api_type=azure_openai_api_type,
        azure_endpoint=azure_openai_embdng_endpoint,
        azure_deployment=azure_openai_embedding_deployment_name,        
        )

def get_llm(temperature=0.3, max_tokens=None, model_name="gpt-4o", metadata=None):
    llm = AzureChatOpenAI(
        model=model_name, temperature=temperature, max_tokens=max_tokens,
        api_key=azure_openai_llm_api_key,
        openai_api_type=azure_openai_api_type,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_llm_endpoint,
        azure_deployment=azure_openai_deployment_name,
        metadata=metadata
        )
    
    return llm

# VectorDB Connections
doc_vector_search = MongoDBAtlasVectorSearch.from_connection_string(
                vec_db_connection_string,
                db_name + "." + doc_collection_name,
                embedding_model,
                index_name=vector_index_name
            )

startup_vector_search = MongoDBAtlasVectorSearch.from_connection_string(
                vec_db_connection_string,
                db_name + "." + startups_collection_name,
                embedding_model,
                index_name=vector_index_name
            )

doc_retriever = doc_vector_search.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 10}
)

startup_retriever = startup_vector_search.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 10}
)

async def get_context(retriever, query: str):
    docs = await retriever.ainvoke(query)
    
    return docs

async def run_rag_chain(prompt, query, context, temperature=0.3):
    llm = get_llm(temperature=temperature)
    
    rag_chain = (
        {"query": RunnablePassthrough(), "context": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        res = await rag_chain.ainvoke({"query": query, "context": context})
    
        if res is None:
            # Error from Azure
            return "An error has occurred. Please try again."
        
        return res
    except ValueError:
        # Error from Azure
        return "An error has occurred. Please try again."
    

async def get_response(query: str, temperature=0.3):
    context = await get_context(doc_retriever, query)
    
    response = await run_rag_chain(
        prompt=prompts.rag_prompt,
        query=query,
        context=context,
        temperature=temperature
        )
    
    return response

async def get_competitor(query: str, temperature=0.3):
    context = await get_context(startup_retriever, query)
    
    response = await run_rag_chain(
        prompt=prompts.rag_prompt,
        query=query,
        context=context,
        temperature=temperature
        )
    
    return response
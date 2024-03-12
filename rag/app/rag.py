#!/usr/bin/python3

import os

import chromadb

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain import hub

#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import ConfluenceLoader

from langchain_openai import AzureChatOpenAI

from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex


OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL')

PROMPT_TEMPLATES = {
    'mistral': """
        <s> [INST]
        You are an retrieval augmented generation assistant.
        The provided context contains documents related to the question.
        Use the following documents only to answer the question.
        All documents containing relevant pieces of the answer must be referenced by source and page.
        If you don't know the answer, just say that you don't know.
        [/INST] </s>
        [INST]
        Context: {context}
        Question: {question}
        Answer:
        References:
        [/INST]
        """,
    'llama2': """
        [INST] <<SYS>>
        Answer the users question only taking into account the following context.
        If the user asks for information not found in the below context, do not answer.
        All context documents containing relevant pieces of the answer must be referenced by source and page.

        <context>
        {context}
        </context>
        <</SYS>>
        
        {question}
        [/INST]
        """,
    'gpt-3.5-turbo': """
        System:
        You are an retrieval augmented generation assistant. 
        The provided context contains documents related to the question.
        Use the document property page_content only to answer the question. 
        All documents containing relevant pieces of the answer must be referenced with the metadata properties source and page, use format [source, page].
        If you don't know the answer, just say that you don't know. 

        Context documents:
        {context} 

        Question:
        {question} 

        Answer:
        
        References:

        """,
    'gpt-4': """
        System:
        You are an retrieval augmented generation assistant. 
        The provided context contains documents related to the question.
        Use the document property page_content only to answer the question. 
        All documents containing pieces of the answer must be referenced with the metadata properties source and page, use format [source, page].
        If you don't know the answer, just say that you don't know. 

        Context documents:
        {context} 

        Question:
        {question} 

        Answer with references:

        """,
    'gpt-4-turbo-preview': """
        System:
        You are an retrieval augmented generation assistant. 
        The provided context contains documents related to the question.
        Use the document property page_content only to answer the question. 
        All documents containing pieces of the answer must be referenced with the metadata properties source and page, use format [source, page].
        If you don't know the answer, just say that you don't know. 

        Context documents:
        {context} 

        Question:
        {question} 

        Answer with references:

        """
}

CHROMA_HOST = os.environ.get('CHROMA_HOST')
CHROMA_PORT = os.environ.get('CHROMA_PORT')

FASTEMBED_CACHE_PATH = os.environ.get('CHROMA_PORT')

LANGCHAIN_TRACING_V2 = os.environ['LANGCHAIN_TRACING_V2']
LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']
LANGCHAIN_ENDPOINT = os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']

OPENAI_ENABLED = os.environ["OPENAI_ENABLED"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]

ATLASSIAN_ENABLED = os.environ["ATLASSIAN_ENABLED"]
ATLASSIAN_URL = os.environ["ATLASSIAN_URL"]
ATLASSIAN_USERNAME = os.environ["ATLASSIAN_USERNAME"]
ATLASSIAN_API_KEY = os.environ["ATLASSIAN_API_KEY"]

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]


class RagBuilder:
    def __init__(self, debug_print_func):
        self.debug = debug_print_func
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.azure_search_index_client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY))
    
    def __get_embedding(self):
        # see model list: https://qdrant.github.io/fastembed/examples/Supported_Models/
        #return FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5') # max 384 tokens
        return FastEmbedEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2') # max 768 tokens
        #return FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large') # max 1024 tokens

    def list_knowledge_base_servers(self):
        return ['local-chroma-db', 'azure-ai-search']

    def list_knowledge_bases(self, knowledge_base_server_name: str):
        if knowledge_base_server_name == 'local-chroma-db':
            collections = self.chroma_client.list_collections()
            return map(lambda c: c.name, collections)
        if knowledge_base_server_name == 'azure-ai-search':
            indexes = self.azure_search_index_client.list_index_names()
            return indexes
        return []

    def get_knowledge_base_details(self, knowledge_base_server_name: str, knowledge_base_name: str):
        if knowledge_base_server_name == 'local-chroma-db':
            collection = self.chroma_client.get_collection(knowledge_base_name)
            #details = vars(collection)
            details = {}
            details['chunk-count'] = collection.count()
            return details
        if knowledge_base_server_name == 'azure-ai-search':
            index = self.azure_search_index_client.get_index_statistics(index_name=knowledge_base_name)
            #self.debug('azure search index statistics: ', index)
            return index
        return {}

    def create_knowledge_base(self, knowledge_base_server_name: str, knowledge_base_name: str):
        if knowledge_base_server_name == 'local-chroma-db':
            collection = self.chroma_client.get_or_create_collection(knowledge_base_name)
        elif knowledge_base_server_name == 'azure-ai-search':
            #index = self.azure_search_index_client.create_index(SearchIndex(name=knowledge_base_name, fields=List[SearchField]))
            AzureSearch(azure_search_endpoint=AZURE_SEARCH_ENDPOINT, azure_search_key=AZURE_SEARCH_KEY, index_name=knowledge_base_name, embedding_function=self.__get_embedding())

    def delete_knowledge_base(self, knowledge_base_server_name: str, knowledge_base_name: str):
        if knowledge_base_server_name == 'local-chroma-db':
            self.chroma_client.delete_collection(knowledge_base_name)
        elif knowledge_base_server_name == 'azure-ai-search':
            self.azure_search_index_client.delete_index(SearchIndex(knowledge_base_name))

    def list_model_servers(self):
        return ['local_ollama', 'openai', 'azure-openai']

    def list_models(self, model_server_name: str):
        if model_server_name == 'local_ollama':
            return ['mistral', 'llama2']
        if model_server_name == 'openai':
            return ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']
        if model_server_name == 'azure-openai':
            return []
        return []

    def build_rag_assistant(self, knowledge_base_server_name: str, knowledge_base_name: str, model_server_name: str, model_name: str):
        vector_store = None
        model = None
        temp = 1

        if knowledge_base_server_name == 'local-chroma-db':
            vector_store = Chroma(client=self.chroma_client, collection_name=knowledge_base_name, embedding_function=self.__get_embedding())
        elif knowledge_base_server_name == 'azure-ai-search':
            vector_store = AzureSearch(azure_search_endpoint=AZURE_SEARCH_ENDPOINT, azure_search_key=AZURE_SEARCH_KEY, index_name=knowledge_base_name, embedding_function=self.__get_embedding())

        if model_server_name == 'local_ollama':
            model = ChatOllama(model=model_name, temperature=temp, base_url=OLLAMA_BASE_URL)
        elif model_server_name == 'openai':
            model = ChatOpenAI(model_name=model_name, temperature=temp, openai_api_key=OPENAI_API_KEY, openai_organization=OPENAI_ORG_ID)

        prompt = PromptTemplate.from_template(PROMPT_TEMPLATES[model_name])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        return RagAssistant(self.debug, vector_store, model, prompt, text_splitter)


class RagAssistant:

    def __init__(self, debug_print_func, vector_store, model, prompt, text_splitter):
        self.debug = debug_print_func
        self.vector_store = vector_store
        self.model = model
        self.prompt = prompt
        self.text_splitter = text_splitter

    def add_pdf(self, collection_name: str, pdf_file_path: str, original_name: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        for doc in docs:
            doc.metadata['source'] = original_name
        self.debug('loaded docs', docs)
        self.__add_docs(docs)

    def add_confluence_page(self, collection_name: str, conf_space_key: str):
        loader = ConfluenceLoader(url=ATLASSIAN_URL, username=ATLASSIAN_USERNAME, api_key=ATLASSIAN_API_KEY)
        docs = loader.load(space_key=conf_space_key, limit=50)
        self.debug(f'loaded docs from confluence space {conf_space_key}', docs)
        self.__add_docs(docs)

    def __add_docs(self, docs):
        chunks = self.text_splitter.split_documents(docs)
        self.debug('splitted chunks', chunks)
        chunks = filter_complex_metadata(chunks)
        self.debug('filtered chunks', chunks)
        self.vector_store.add_documents(documents=chunks)

    def ask(self, query: str):
        retriever = self.vector_store.as_retriever(
            #search_type = "similarity_score_threshold", search_kwargs = {"k": 3,"score_threshold": 0.5}
            search_type = "similarity_score_threshold", search_kwargs = {"k": 6,"score_threshold": 0.3}
            #search_type = "mmr", search_kwargs = {'k': 5, 'fetch_k': 50}
        )
        chain = ({"context": retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        return chain.invoke(query)

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

from langchain_community.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI


OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL')
OLLAMA_MODELS = {
    #'mistral': """
    #    <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
    #    to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
    #     maximum and keep the answer concise.[/INST] </s> 
    #    [INST] Question: {question} 
    #    Context: {context} 
    #    Answer: [/INST]
    #    """,
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
        [INST] <<SYS>>Answer the users question only taking into account the following context. 
        If the user asks for information not found in the below context, do not answer.

        <context>
        {context}
        </context>
        <</SYS>>

         {question} [/INST]
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
        All documents containing relevant pieces of the answer must be referenced with the metadata properties source and page, use format [source, page].
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

LANGCHAIN_TRACING_V2 = os.environ['LANGCHAIN_TRACING_V2']
LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']
LANGCHAIN_ENDPOINT = os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']

OPENAI_ENABLED = os.environ["OPENAI_ENABLED"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]

if OPENAI_ENABLED != "true":
    OLLAMA_MODELS = {k: v for k, v in OLLAMA_MODELS.items() if not k.startswith('gpt')}

class ChatAssistant:

    def __init__(self, debug_print_func, model_name):
        self.debug = debug_print_func
        if model_name.startswith('gpt'):
            self.model = ChatOpenAI(model_name=model_name, temperature=0.2,
                openai_api_key=OPENAI_API_KEY, openai_organization=OPENAI_ORG_ID)
        else:
            self.model = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.prompt = PromptTemplate.from_template(OLLAMA_MODELS[model_name])

    def ingest(self, collection_name: str, pdf_file_path: str, original_name: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        for doc in docs:
            doc.metadata['source'] = original_name
        self.debug('loaded docs', docs)
        chunks = self.text_splitter.split_documents(docs)
        self.debug('splitted chunks', chunks)
        chunks = filter_complex_metadata(chunks)
        self.debug('filtered chunks', chunks)
        vector_store = self.__get_vector_store(collection_name)
        vector_store.add_documents(documents=chunks)

    def __get_vector_store(self, collection_name: str):
        collection = self.chroma_client.get_or_create_collection(collection_name)
        vector_store = Chroma(client=self.chroma_client, collection_name=collection_name, 
            embedding_function=FastEmbedEmbeddings())
        #self.debug('vector_store.get()', vector_store.get())
        return vector_store

    def __get_chain(self, vector_store):
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        return ({"context": retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, collection_name: str, query: str):
        vector_store = self.__get_vector_store(collection_name)
        chain = self.__get_chain(vector_store)
        return chain.invoke(query)

    def list_collections(self):
        collections = self.chroma_client.list_collections()
        return map(lambda c: c.name, collections)

    def get_collection_details(self, collection_name: str):
        collection = self.chroma_client.get_collection(collection_name)
        details = vars(collection)
        details['count'] = collection.count()
        return details

    def create_collections(self, collection_name: str):
        collection = self.chroma_client.get_or_create_collection(collection_name)

    def delete_collections(self, collection_name: str):
        self.chroma_client.delete_collection(collection_name)

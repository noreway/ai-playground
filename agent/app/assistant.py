#!/usr/bin/python3

# https://python.langchain.com/docs/modules/agents/agent_types/openai_assistants


import os

from langchain.agents import AgentExecutor
from langchain.agents.openai_assistant import OpenAIAssistantRunnable


LANGCHAIN_TRACING_V2 = os.environ['LANGCHAIN_TRACING_V2']
LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']
LANGCHAIN_ENDPOINT = os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']

OPENAI_ENABLED = os.environ["OPENAI_ENABLED"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]
OPENAI_ASSISTANT_ID = os.environ["OPENAI_ASSISTANT_ID"]


class AddressAssistantAI:
    def __init__(self):
        self.agent = OpenAIAssistantRunnable(assistant_id=OPENAI_ASSISTANT_ID, as_agent=True)
        self.tools = []
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def ask(self, query: str):
        result = self.agent_executor.invoke({"content": query})
        return result["output"]

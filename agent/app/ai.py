#!/usr/bin/python3

# tutorial used to adapt lightening use case
# https://python.langchain.com/docs/modules/agents/how_to/custom_agent

# may be for later prototypes
# https://python.langchain.com/docs/use_cases/sql/


import os
import requests

from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor


LANGCHAIN_TRACING_V2 = os.environ['LANGCHAIN_TRACING_V2']
LANGCHAIN_PROJECT = os.environ['LANGCHAIN_PROJECT']
LANGCHAIN_ENDPOINT = os.environ['LANGCHAIN_ENDPOINT']
LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']

OPENAI_ENABLED = os.environ["OPENAI_ENABLED"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]


@tool
def set_shelly_light_on(ip: str):
    """Sets Shelly controlled light on."""
    response = requests.get(f'http://{ip}/relay/0?turn=on')
    return response.json()

@tool
def set_shelly_light_off(ip: str):
    """Sets Shelly controlled light off."""
    response = requests.get(f'http://{ip}/relay/0?turn=off')
    return response.json()


class IlluminatingAI:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)   # or 'gpt-4', 'gpt-4-turbo-preview'
        self.tools = [set_shelly_light_on, set_shelly_light_off]
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "You are very powerful light assistant. Use your tools to turn shelly controlled lights on or off."
            ),
            (
                "user", 
                "There are lights named by ip address 192.168.11.210 (south west), 192.168.11.212 (south east), 192.168.11.213 (north east), "
                + "192.168.11.214 (north), 192.168.11.215 (north west), 192.168.11.216 (west) and 192.168.11.217 (east)."
            ),
            (
                "assistant", 
                "Thank you for the list. Now I can turn those lights on and off using my tools and the ip adressed."
            ),
            (
                "user", 
                "{input}"
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def ask(self, query: str):
        return self.agent_executor.invoke({"input": query})["output"]


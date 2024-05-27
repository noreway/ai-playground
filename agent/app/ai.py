#!/usr/bin/python3

# tutorial used to adapt lightening use case
# https://python.langchain.com/docs/modules/agents/how_to/custom_agent

# may be for later prototypes
# https://python.langchain.com/docs/use_cases/sql/


import os
import requests
import json
from time import sleep

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain import hub
from langchain.agents import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from langchain_openai import ChatOpenAI, OpenAI


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

@tool
def wait(seconds: float):
    """Waits for the provided seconds an then returns."""
    sleep(seconds)

@tool
def do_http_get_request(url: str):
    """Sends a HTTP GET request using python requests and returns the response object."""
    response = requests.get(url)
    if response.status_code == 200:
        return { "status_code": response.status_code, "text": response.text }
    else:
        return { "status_code": response.status_code }


class IlluminatingAI:
    def __init__(self):
        # models: 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview'
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        #self.tools = [set_shelly_light_on, set_shelly_light_off, wait, do_http_get_request]
        self.tools = [do_http_get_request]
        self.chat_history = []
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "You are very powerful light assistant. Use your tools to turn shelly controlled lights on or off. "
                + "Ensure to use the correct API calls depending on the Shelly switch model generation."
                + "You can use the endpoint /shelly to check the model and generation."
                + "Useful link: https://shelly-api-docs.shelly.cloud/gen2/ComponentsAndServices/Switch#http-endpoint-relayid"
                + "Your should always think about what to do, defining an action to take and observe the result. Your can iterate N times."
            ),
            (
                "user", 
                "There are lights named by ip address 192.168.11.210 (south west), 192.168.11.212 (south east), 192.168.11.213 (north east), "
                + "192.168.11.214 (north), 192.168.11.215 (north west), 192.168.11.216 (west) and 192.168.11.217 (east). "
            ),
            (
                "assistant", 
                "Thank you for the list. Now I can turn those lights on and off using my tools and the ip adressed."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
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
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def ask(self, query: str):
        result = self.agent_executor.invoke({
            "input": query, 
            "chat_history": self.chat_history
        })
        self.chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=result["output"])
        ])
        return result["output"]

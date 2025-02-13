# This is going to a streamlit app that will use the AI agents to generate content.
# Using Semantic Kernel and Azure OpenAI. Streamlit will be used to create the UI.
# Semantic Kernel Agents will need to be able to search the web for clothing items.
# The content will be prompted by the user telling us where they are going, and what time of the year.
# From there our agents will look up the weather for that location, and then recommend what to wear.
# The agents will also look up specific clothing items that are available for purchase, and present them to the user.
# Lastly, the agents will display the cost of the clothing items, and the total cost.
# We are going to have 5 agents
# 1. Weather Agent
# 2. Clothing Agent
# 3. Shopping Agent
# 4. Display Agent
# 5. Cost Agent

import asyncio
import os
import yaml
import streamlit as st

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments, KernelParameterMetadata, KernelPlugin
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.connectors.search.bing import BingSearch
from semantic_kernel.filters.functions.function_invocation_context import FunctionInvocationContext
from semantic_kernel.connectors.search_engine import BingConnector, GoogleConnector
from semantic_kernel.core_plugins import WebSearchEnginePlugin

async def main():
    # First we start by creating the streamlit app.
    st.set_page_config(page_title="👨‍🏫 Content Agents", layout="wide")
    st.title('👨‍🏫 Content Agents')

    # Create the sidebar, which will contain all the keys.
    st.sidebar.title('🔑 Application Keys')
    azure_openai_key = st.sidebar.text_input('Azure OpenAI Key', '')
    azure_openai_endpoint = st.sidebar.text_input('Azure OpenAI Endpoint', '')
    azure_openai_version = st.sidebar.text_input('Azure OpenAI Version', '2024-05-01-preview')
    azure_openai_deployment_name = st.sidebar.text_input('Azure OpenAI Deployment Name', 'gpt-4o')
    bing_api_key = st.sidebar.text_input('Bing API Key', '')
    
    # Define the Kernel
    kernel = Kernel()

    # Add Azure OpenAI Connector
    chat_completion_service = AzureChatCompletion(
        deployment_name=azure_openai_deployment_name,  
        api_key=azure_openai_key,
        endpoint=azure_openai_endpoint, # Used to point to your service
        api_version=azure_openai_version, # Used to point to your service
    )

    connector = BingConnector(bing_api_key)
    kernel.add_plugin(WebSearchEnginePlugin(connector), "WebSerarch")
    kernel.add_service(chat_completion_service)

    arguments = KernelArguments(
        settings=PromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto(),)
    )

    # Now we create the form, which will contain the text input, as well as the submit button.
    with st.form('ai_form'):
        text = st.text_area('Subject:', 'What is the recommendation for what to wear in Seattle in the spring for an adult male?')

        submit = st.form_submit_button('Submit')
        if submit:
            response = await kernel.invoke_prompt(text, arguments=arguments)
            #result = await kernel.invoke(web_plugin["search"], query=text)
            st.write(response.value)
            
    
if __name__ == "__main__":
    asyncio.run(main())
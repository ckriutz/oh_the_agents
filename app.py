import asyncio
import os
import yaml
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from output_classes import SocialMediaPost, ContentOutput
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

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
    serper_api_key = st.sidebar.text_input('Serper API Key', '')
    
    # Required
    os.environ['AZURE_OPENAI_API_KEY'] = azure_openai_key
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_openai_endpoint
    os.environ['AZURE_API_VERSION'] = azure_openai_version
    os.environ['SERPER_API_KEY'] = serper_api_key

    # Define file paths for YAML configurations
    files = {
        'agents': 'config/agents.yaml',
        'tasks': 'config/tasks.yaml'
    }

    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    agents_config = configs['agents']
    tasks_config = configs['tasks']

    websearchtool = WebsiteSearchTool(
        config=dict(
            llm=dict(
                provider="azure_openai", # or google, openai, anthropic, llama2, ...
                config=dict(
                    model="gpt-4o-mini",
                    # temperature=0.5,
                    # top_p=1,
                    # stream=true,
                ),
            ),
            embedder=dict(
                provider="azure_openai", # or openai, ollama, ...
                config=dict(
                    model="text-embedding-ada-002",
                    deployment_name= "text-embedding-ada-002",
                    #task_type="retrieval_document",
                    # title="Embeddings",
                ),
            ),
        )
    )

    # Creating Agents
    market_news_monitor_agent = Agent(
        config=agents_config['market_news_monitor_agent'],
        tools=[SerperDevTool(), ScrapeWebsiteTool()],
        llm=LLM(model=f'azure/gpt-4o'),
    )

    data_analyst_agent = Agent(
        config=agents_config['data_analyst_agent'],
        tools=[SerperDevTool(), websearchtool],
        llm=LLM(model=f'azure/gpt-4o'),
    )

    content_creator_agent = Agent(
        config=agents_config['content_creator_agent'],
        tools=[SerperDevTool(), websearchtool], # had to customize WebsiteSearchTool
        llm=LLM(model=f'azure/gpt-4o')
    )

    quality_assurance_agent = Agent(
        config=agents_config['quality_assurance_agent'],
        llm=LLM(model=f'azure/gpt-4o')
    )

    # Creating Tasks
    monitor_financial_news_task = Task(
        config=tasks_config['monitor_financial_news'],
        agent=market_news_monitor_agent
    )

    analyze_market_data_task = Task(
        config=tasks_config['analyze_market_data'],
        agent=data_analyst_agent
    )

    create_content_task = Task(
        config=tasks_config['create_content'],
        agent=content_creator_agent,
        context=[monitor_financial_news_task, analyze_market_data_task]
    )

    quality_assurance_task = Task(
        config=tasks_config['quality_assurance'],
        agent=quality_assurance_agent,
        output_pydantic=ContentOutput
    )

    # Creating Crew
    content_creation_crew = Crew(
        agents=[
            market_news_monitor_agent,
            data_analyst_agent,
            content_creator_agent,
            quality_assurance_agent
        ],
        tasks=[
            monitor_financial_news_task,
            analyze_market_data_task,
            create_content_task,
            quality_assurance_task
        ],
        verbose=True
    )

    # Assign loaded configurations to specific variables
    agents_config = configs['agents']
    tasks_config = configs['tasks']

    # Now we create the form, which will contain the text input, as well as the submit button.
    with st.form('ai_form'):
        text = st.text_area('Subject:', 'The effect of inflation on the markets in 2024.')

        submit = st.form_submit_button('Submit')
        if submit:
            result = content_creation_crew.kickoff(inputs={'subject': text})
            st.write(result)
            
    
if __name__ == "__main__":
    asyncio.run(main())
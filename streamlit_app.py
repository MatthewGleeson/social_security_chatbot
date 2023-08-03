import streamlit as st
import os
import pinecone
import openai

import datetime
import json
import os
import pandas as pd
import re
from typing import List, Union
import zipfile


from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
# LLM wrapper
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
# Conversational memory
from langchain.memory import ConversationBufferWindowMemory
# Embeddings and vectorstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

st.title('Social Security Chatbot')

api_key = os.getenv("PINECONE_API_KEY")

# find environment next to your API key in the Pinecone console
env =  "us-central1-gcp"

pinecone.init(api_key=api_key, environment=env)
index_name = "social-security"
index = pinecone.Index(index_name=index_name)


# Configuring the embeddings to be used by our retriever to be OpenAI Embeddings, matching our embedded corpus
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


# Loads a docsearch object from an existing Pinecone index so we can retrieve from it
docsearch = Pinecone.from_existing_index(index_name,embeddings,text_key='text_chunk')
retriever = docsearch.as_retriever()
retrieval_llm = OpenAI(temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))
social_security_retriever = RetrievalQA.from_chain_type(llm=retrieval_llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

# Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
template = """Answer the following questions as best you can, speaking as if you are a nurse with extensive medical knowledge giving advice to a patient. You have access to the following tools:

{tools}

Use the following format and never deviate from this structure. The structure before each colon is also imperative to include:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. This answer should consolidate the chain of thought from earlier and explain the answer in a clear way that could be understood by a 7th grader. The answer should be phrased as if speaking directly to the patient. If you don't know the answer to the question, return the final answer as "Final Answer: I don't know the answer to this question"

Begin!

Question: {input}
{agent_scratchpad}"""

#right now we only have a single tool which is the vector database

#right now we only have a single tool which is the vector database
from langchain.tools import BaseTool

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class KnowledgeBaseWithSources(BaseTool):
    name = "Knowledge Base"
    description= "Useful for getting answers for social security-related questions, as well as information on diseases that qualify for social security. Input should be a fully formed question."
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        response = social_security_retriever({"query": query})
        print(str(response))
        returned_links.append(response["source_documents"])
        return response["result"]
    
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


tools = [
    KnowledgeBaseWithSources(),
]

class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)

        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)




class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

# Initiate our LLM - default is 'gpt-3.5-turbo'
llm = ChatOpenAI(temperature=0,openai_api_key=os.environ["OPENAI_API_KEY"])

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    # We use "Observation" as our stop sequence so it will stop when it receives Tool output
    # If you change your prompt template you'll need to adjust this as well
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

st.write("Example question: I have a child with down syndrome. What do I need to know to determine if he qualifies for social security?")

form = st.form(key='my-form')
chatbot_query = form.text_area('Ask the Social Security Chatbot a question!')

chatbot_access_token = form.text_input('Access Token for the chatbot')
submit = form.form_submit_button('Submit')

offline_testing = False

if submit:
    returned_links = []
    if chatbot_access_token!= os.getenv("CHATBOT_ACCESS_TOKEN"):
        st.write("Sorry, you have provided an invalid access token. Please check it is correct and try again, or reach out to the creator to get access")
    else:
        if offline_testing:
            st.write(chatbot_query)
        else:
            output = agent_executor.run(chatbot_query)
            st.success(f"**:blue[Chatbot response]:** {output}")
            st.info(f"I read this document to create my response: {returned_links[0][0].metadata['url']}")
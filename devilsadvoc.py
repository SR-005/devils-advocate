import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

from duckduckgo_search import DDGS



load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature="0.2")

@tool       #a decorator: 
def searchweb(query: str) -> str:
    #this is a docstring. It will be read by langchain inorder to understand about what this tool does
    """Search the web for counterarguments, research papers, statistics or criticisms opposing a claim """     

    print("Searching the Web...\n")
    with DDGS() as ddgs:
        results=[r['body']for r in ddgs.text(query, max_results=3)]     #take all the search results from web and append it as a list
        return "\n".join(results) if results else "No Opposing Evidence Found!"     #joins all the result string into one string and returns it

#Package the tools into a list
tools=[searchweb]
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

#agent fills the placeholders and reads this template when a new input is recieved or even multiple times in a single execution
prompt=ChatPromptTemplate.from_messages([
    #foundational system intruction about how it should behave
    ("system", """You are an analytical Red Team Agent and ruthless Devil's Advocate.       
    Your mission is to rigorously challenge the user's opinions, technical assumptions, or startup ideas.
    Rules:
    1. Never blindly agree or validate the user's thesis.
    2. For any claim made by the user, use the `search_web` tool to find counter-data, failure case studies, bottlenecks, or opposing research.
    3. Synthesize the findings into a clear, sharp, evidence-backed counterargument. Citing specific stats or facts discovered during your search is mandatory."""),
    ("placeholder","{chathistory}"),        #holds prior conversation. chat history is injected into this slot
    ("human",{input}),                      #user live message is injected to this place
    ("placeholder", "{agentscratchpad}"),   #where LangChain writes the agent's internal monologue. the workflow and decision by agent is made here.
])

#this tells the gemini ai what all tools are available and will be attached to our prompts
agent=create_tool_calling_agent(llm, tools, prompt)

#creating a runtime executor, verbose=True lets us watch the internal thinking process and tool selection
agentexecutor=AgentExecutor(agent=agent,tools=tools,verbose=True)

if __name__=="__main__":
    print("--------Devils Advocate--------")

    #the statement ai will oppose
    thesis="Python is strictly the best programming language for all backend development."
    print(f"User Claim: {thesis}\n")

    result=agentexecutor.invoke({
        "input": thesis,
        "chathistory": []
    })

    print("\n--- Counterargument ---")
    print(result["output"])



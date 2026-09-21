import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.tools import tool
from langchain.agents import create_agent

from ddgs import DDGS


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)

@tool  
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
prompt="""You are an analytical Red Team Agent and ruthless Devil's Advocate.       
    Your mission is to rigorously challenge the user's opinions, technical assumptions, or startup ideas.
    Rules:
    1. Never blindly agree or validate the user's thesis.
    2. For any claim made by the user, use the `searchweb` tool to find counter-data, failure case studies, bottlenecks, or opposing research.
    3. Synthesize the findings into a clear, sharp, evidence-backed counterargument. Citing specific stats or facts discovered during your search is mandatory."""

#this tells the gemini ai what all tools are available and will be attached to our prompts
devilagent=create_agent(model=llm, tools=tools, system_prompt=prompt)


if __name__=="__main__":
    print("--------Devils Advocate--------")

    #the statement ai will oppose
    thesis="Python is strictly the best programming language for all backend development."
    print(f"User Claim: {thesis}\n")

    result=devilagent.invoke({
        "messages":[{
            "role": "user",
            "content": thesis
        }],
    })

    print("\n--- Counterargument ---")
    response=result["messages"][-1].content
    response=response[0]["text"] if isinstance(response,list) else response   #extract clean text from response if it is a list, else return same
    print(response)


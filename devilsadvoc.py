import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from ddgs import DDGS


load_dotenv()
app = FastAPI(title="Devil's Advocate API")

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
devilagent=create_agent(model=llm, 
                        tools=tools, 
                        checkpointer=MemorySaver(),         #for memory management
                        system_prompt=prompt)

class ChatRequest(BaseModel):
    message: str
    threadid: str

@app.post("/api/chat")
async def chat(payload: ChatRequest):
    threadconfig={"configurable": {"thread_id": "1"}}        #creating the session thread for memory management
    result=devilagent.invoke({
                "messages":[HumanMessage(content=userinput)],
            },
            config=threadconfig) 
    response=result["messages"][-1].content
    response=response[0]["text"] if isinstance(response,list) else response   #extract clean text from response if it is a list, else return same

app.mount("/static", StaticFiles(directory="static"), name="static")        #mount static files

@app.get("/")
async def root():
    return FileResponse("static/page.html")

if __name__=="__main__":
    uvicorn.run("server.py:app", host="127.0.0.1", port=8000, reload=True)

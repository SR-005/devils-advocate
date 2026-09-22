import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from ddgs import DDGS


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2) #gemini-3.6-flash

@tool  
def searchweb(query: str) -> str:
    #this is a docstring. It will be read by langchain inorder to understand about what this tool does
    """Search the web for counterarguments, research papers, statistics or criticisms opposing a claim """     

    print("Searching the Web...\n")
    try:
        with DDGS(timeout=7) as ddgs:
            results=[r['body']for r in ddgs.text(query, max_results=3)]     #take all the search results from web and append it as a list
            validresults=[r for r in results if r]
            if validresults:
                return "\n".join(validresults)     #joins all the result string into one string and returns it

    except Exception as e:
        print(f"[Tool Error / Timeout]: {e}")

    return (
        "No new web data found. STOP searching immediately. "
        "Proceed directly to synthesizing your final counterargument using your internal knowledge base."
    )

#Package the tools into a list
tools=[searchweb]

#agent fills the placeholders and reads this template when a new input is recieved or even multiple times in a single execution
prompt = """You are an analytical Red Team Agent and ruthless Devil's Advocate.
    Your mission is to rigorously challenge the user's opinions, technical assumptions, or product assertions.

    Strict Execution Rules:
    1. Never validate or agree with the user's thesis.
    2. You are allowed EXACTLY ONE call to `searchweb` to gather counter-evidence, competitor benchmarks, or known hardware/software limitations.
    3. Once `searchweb` returns ANY information (or fails), DO NOT call `searchweb` again. You MUST generate your final rebuttal immediately in the next step.
    4. Synthesize findings into a sharp, evidence-backed counterargument detailing specific specs, engineering trade-offs, and competitor strengths."""

#this tells the gemini ai what all tools are available and will be attached to our prompts
devilagent=create_agent(model=llm, 
                        tools=tools, 
                        checkpointer=MemorySaver(),         #for memory management
                        system_prompt=prompt)


if __name__=="__main__":

    threadconfig={"configurable": {"thread_id": "1"},
                  "recursion_limit": 8}        #creating the session thread for memory management

    print("--------Devils Advocate--------")

    while True:
        #the statement ai will oppose
        userinput=input("You: ").strip()

        if not userinput:
            continue
        if userinput.lower() in ["exit","quit","q"]:            #chatbot look break logic
            break

        print(f"User Claim: {userinput}\n")

        result=devilagent.invoke({
            "messages":[HumanMessage(content=userinput)],
        },
        config=threadconfig)  

        print("\n--- Counterargument ---")
        response=result["messages"][-1].content
        response=response[0]["text"] if isinstance(response,list) else response   #extract clean text from response if it is a list, else return same
        print(response)

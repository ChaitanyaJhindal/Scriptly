import os
import subprocess
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain.schema import SystemMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def run_command(cmd: str):
    """
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"Output:\n{result.stdout}\n"
        if result.stderr:
            output += f"Error:\n{result.stderr}\n"
        return output
    except Exception as e:
        return f"Error executing command: {str(e)}"

def create_chat_graph(checkpointer):
    llm = init_chat_model(
        model_provider="openai", model="gpt-4"
    )
    llm_with_tool = llm.bind_tools(tools=[run_command])
    
    def chatbot(state: State):
        system_prompt = SystemMessage(content="""
            You are an AI Coding assistant who takes an input from user and based on available
            tools you choose the correct tool and execute the commands.
                                      
            You can even execute commands and help user with the output of the command.

            IMPORTANT: When responding, structure your response in TWO PARTS:
            1. SPEECH_OUTPUT: A brief, conversational summary (max 2-3 sentences)
            2. DETAILED_RESPONSE: The complete technical details
            
            Format your response like this:
            SPEECH_OUTPUT: [Brief summary for voice output]
            
            DETAILED_RESPONSE: [Complete technical details]

            CRITICAL WORKFLOW FOR FILE CREATION:
            When user asks for ANY code/program creation, you MUST:
            
            1. ALWAYS use run_command tool to create the file with PowerShell syntax
            2. ALWAYS use run_command tool to verify the file was created (ls command)
            3. ALWAYS use run_command tool to display the file content (cat or Get-Content)
            4. Provide execution instructions
            
            MANDATORY SEQUENCE - You must execute these commands in order:
            
            Step 1 - Create file:
            @"
            [ACTUAL CODE CONTENT HERE]
            "@ | Out-File -FilePath "filename.ext" -Encoding UTF8
            
            Step 2 - List directory to verify:
            ls
            
            Step 3 - Display file content:
            cat filename.ext
            
            EXAMPLE WORKING PATTERN:
            User: "create a calculator in python"
            You must:
            1. Use run_command to create calculator.py with the code
            2. Use run_command to list directory (ls)
            3. Use run_command to show file content (cat calculator.py)
            4. Provide execution instructions
            
            RULES:
            1. NEVER just describe - ALWAYS use run_command tool for file operations
            2. Use the @"..."@ syntax for multi-line content
            3. Save files to current directory (no full path needed)
            4. Use appropriate file extensions (.html, .cpp, .py, .js, .css)
            5. Always execute the 3-step sequence above
            6. Be specific and direct in your tool calls
            
            EXECUTION COMMANDS BY FILE TYPE:
            - Python (.py): python filename.py
            - C++ (.cpp): g++ filename.cpp -o filename.exe && filename.exe
            - HTML (.html): start filename.html
            - JavaScript (.js): node filename.js
            
            You have access to my terminal and can save code in the actual directory. 
            You must use run_command tool for every file creation request. Do not just provide instructions.
        """)

        message = llm_with_tool.invoke([system_prompt] + state["messages"])
        return {"messages": [message]}
    
    tool_node = ToolNode(tools=[run_command])

    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile(checkpointer=checkpointer)

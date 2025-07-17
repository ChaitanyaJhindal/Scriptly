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
    original_query: str
    processed_query: str
    query_confirmed: bool
    is_simple_query: bool

@tool
def run_command(cmd: str):
    """
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    try:
        # Use subprocess to capture output properly
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
    # Initialize LLM here, after environment variables are loaded
    llm = init_chat_model(
        model_provider="openai", model="gpt-4o-mini",
    )
    llm_with_tool = llm.bind_tools(tools=[run_command])
    
    def query_judge(state: State):
        """Judge if the query is simple and straightforward"""
        original_query = state["messages"][-1].content
        
        judge_prompt = SystemMessage(content=f"""
        You are a query complexity judge. Analyze the user's request and determine if it's simple and straightforward enough to execute directly without further processing.
        
        SIMPLE QUERIES (execute directly):
        - Clear, specific requests for creating applications/websites
        - Direct commands like "create a calculator", "make a todo app", "build a website"
        - Requests with clear programming language specified
        - Simple file operations
        - Running existing programs/code
        
        COMPLEX QUERIES (need processing):
        - Vague or ambiguous requests
        - Missing important details
        - Multiple unclear requirements
        - Conflicting instructions
        - Very broad requests without specifics
        
        User's query: "{original_query}"
        
        Analyze this query:
        1. Is it clear what the user wants?
        2. Are the requirements specific enough?
        3. Can this be executed directly without clarification?
        
        Respond with only: SIMPLE or COMPLEX
        """)
        
        judge_message = llm.invoke([judge_prompt])
        is_simple = "SIMPLE" in judge_message.content.upper()
        
        return {
            "messages": state["messages"],
            "original_query": original_query,
            "processed_query": original_query if is_simple else "",
            "query_confirmed": is_simple,
            "is_simple_query": is_simple
        }
    
    def query_processor(state: State):
        """Process and clarify the user's query using chain of thought"""
        if state.get("is_simple_query", False) or state.get("query_confirmed", False):
            return state  # Skip if simple or already confirmed
            
        original_query = state["messages"][-1].content
        
        processing_prompt = SystemMessage(content=f"""
        You are a query processing assistant. Analyze the user's request using chain of thought reasoning to understand what they really want.
        
        CHAIN OF THOUGHT PROCESS:
        1. UNDERSTAND: What is the user asking for?
        2. ANALYZE: What type of task is this? (coding, web development, file management, etc.)
        3. CLARIFY: What specific actions need to be taken?
        4. ENHANCE: Add any missing details or best practices
        5. STRUCTURE: Create a clear, actionable request
        
        User's original query: "{original_query}"
        
        Please think through this step by step:
        
        THINKING:
        1. Understanding: [What does the user want?]
        2. Analysis: [What category/type of task is this?]
        3. Clarification: [What specific steps are needed?]
        4. Enhancement: [What details should be added for better results?]
        5. Structure: [How should this be organized?]
        
        PROCESSED_QUERY: [Provide a clear, detailed, and actionable version of the user's request]
        
        SPEECH_OUTPUT: I need to clarify a few details about your request.
        """)
        
        processed_message = llm.invoke([processing_prompt])
        
        # Extract the processed query
        content = processed_message.content
        if "PROCESSED_QUERY:" in content:
            processed_query = content.split("PROCESSED_QUERY:")[1].strip()
        else:
            processed_query = original_query
            
        return {
            "messages": state["messages"],
            "original_query": original_query,
            "processed_query": processed_query,
            "query_confirmed": False,
            "is_simple_query": False
        }
    
    def confirmation_handler(state: State):
        """Handle query confirmation"""
        if state.get("is_simple_query", False) or state.get("query_confirmed", False):
            return state  # Skip if simple or already confirmed
            
        confirmation_prompt = SystemMessage(content=f"""
        Present the processed query to the user for confirmation.
        
        Original request: "{state.get('original_query', '')}"
        Processed request: "{state.get('processed_query', '')}"
        
        Format your response as:
        
        SPEECH_OUTPUT: I understand you want [brief summary]. Should I proceed with this?
        
        DETAILED_RESPONSE: 
        Your original request: {state.get('original_query', '')}
        
        My understanding: {state.get('processed_query', '')}
        
        Please confirm if this is what you want me to do, or let me know if I should modify anything.
        """)
        
        confirmation_message = llm.invoke([confirmation_prompt])
        
        return {
            "messages": state["messages"] + [confirmation_message],
            "original_query": state.get("original_query", ""),
            "processed_query": state.get("processed_query", ""),
            "query_confirmed": False,
            "is_simple_query": False
        }
    
    def chatbot(state: State):
        # If query is not confirmed and not simple, don't execute
        if not state.get("query_confirmed", False) and not state.get("is_simple_query", False):
            return state
            
        system_prompt = SystemMessage(content=r"""
            You are an AI Coding assistant who helps users create, modify, and manage code files on their Windows system.
            You can also run programs, websites, and applications when requested.
            
            IMPORTANT: When responding, structure your response in TWO PARTS:
            1. SPEECH_OUTPUT: A brief, conversational summary ONLY for crucial steps (max 2-3 sentences)
            2. DETAILED_RESPONSE: The complete technical details, commands, and code
            
            Format your response like this:
            SPEECH_OUTPUT: [Brief summary for voice output - ONLY for crucial steps]
            
            DETAILED_RESPONSE: [Complete technical details]
            
            SPEECH_OUTPUT GUIDELINES - SPEAK ONLY FOR CRUCIAL STEPS:
            - When starting a new project: "I'm creating a [project type] for you"
            - When running code: "Running your program now"
            - When code is ready: "Your [project type] is ready and running"
            - When errors occur: "There was an issue, let me fix it"
            - For simple file operations: DO NOT SPEAK (leave SPEECH_OUTPUT empty)
            
            AUTOMATIC EXECUTION:
            - If user asks to run code, automatically execute it without asking
            - If user says "run it", "execute it", "start it" - run the most recent program
            - Always run programs after creating them if it's implied or requested
            
            IMPORTANT FILE CREATION GUIDELINES:
            1. Always create the 'chat_gpt' directory first: mkdir chat_gpt
            2. For creating files with content, use this PowerShell method:
               - For single line: echo "content" > filename
               - For multi-line files: Use @" and "@ delimiters
               - Example: @"
                         line1
                         line2
                         "@ | Out-File -FilePath filename -Encoding UTF8
            3. Always verify file creation by listing directory contents: dir chat_gpt
            4. Always provide the complete file path: C:\Users\Chait\Desktop\Cursor(voice)\app\chat_gpt\filename
            
            RUNNING PROGRAMS AND WEBSITES:
            - Run Python files: python chat_gpt\filename.py
            - Run Java files: javac chat_gpt\filename.java && java -cp chat_gpt filename
            - Run C++ files: g++ chat_gpt\filename.cpp -o chat_gpt\filename.exe && chat_gpt\filename.exe
            - Run HTML files: start chat_gpt\filename.html (opens in default browser)
            - Run JavaScript with Node: node chat_gpt\filename.js
            - Start local servers: python -m http.server 8000 (for Python web server)
            - Install packages: pip install package_name or npm install package_name
            - Run batch files: chat_gpt\filename.bat
            
            AUTO-RUN DETECTION:
            - If user says "create and run", "make and execute", "build and start" - automatically run after creating
            - If user mentions "run", "execute", "start" anywhere in request - run the program
            - For web applications - automatically open in browser
            - For servers - automatically start the server
            
            WEB DEVELOPMENT SPECIFIC:
            - Create HTML files with proper structure
            - Include CSS styling in <style> tags or separate files
            - Add JavaScript functionality
            - For local web server: python -m http.server 8000, then open http://localhost:8000
            - For Node.js apps: npm init -y, npm install express, node app.js
            
            WINDOWS COMMANDS TO USE:
            - Show current directory: cd (not pwd)
            - List files: dir (not ls)
            - Create directories: mkdir foldername
            - Create files with content: Use Out-File method for multi-line content
            - Read files: type filename (not cat)
            - Navigate: cd "path with spaces"
            - Run executables: .\filename.exe or just filename.exe
            - Open files: start filename (opens with default program)
            
            STEP-BY-STEP PROCESS for file creation and running:
            1. Create chat_gpt directory: mkdir chat_gpt
            2. Create the file with proper PowerShell syntax
            3. Verify creation: dir chat_gpt
            4. Show complete file path
            5. If requested to run OR if it's implied: Use appropriate run command
            6. For websites: Create HTML/CSS/JS files and open in browser or start server
            7. Optionally read the file to confirm content: type chat_gpt\filename
            
            EXAMPLES of SPEECH_OUTPUT (only for crucial steps):
            - "I'm creating a Python calculator for you" (when starting)
            - "Running your program now" (when executing)
            - "Your todo app is ready and running" (when completed and running)
            - "" (empty - for simple file operations)
            
            Always be thorough and verify your work. If a command fails, try alternative methods.
            When asked to run something, always execute the appropriate command after creating the file.
            Detect run requests automatically and execute without asking for permission.
        """)

        # Use the processed query or original query
        query_to_use = state.get("processed_query") or state["messages"][-1].content
        message = llm_with_tool.invoke([system_prompt, {"role": "user", "content": query_to_use}])
        
        return {"messages": state["messages"] + [message]}
    
    def should_process_query(state: State):
        """Determine if query needs processing"""
        if state.get("is_simple_query", False):
            return "execute"
        return "process" if not state.get("query_confirmed", False) else "execute"
    
    def should_confirm_query(state: State):
        """Determine if query needs confirmation"""
        if state.get("is_simple_query", False):
            return "execute"
        return "confirm" if not state.get("query_confirmed", False) else "execute"
    
    tool_node = ToolNode(tools=[run_command])

    graph_builder = StateGraph(State)

    graph_builder.add_node("query_judge", query_judge)
    graph_builder.add_node("query_processor", query_processor)
    graph_builder.add_node("confirmation_handler", confirmation_handler)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "query_judge")
    graph_builder.add_conditional_edges(
        "query_judge",
        should_process_query,
        {
            "process": "query_processor",
            "execute": "chatbot",
        }
    )
    graph_builder.add_conditional_edges(
        "query_processor",
        should_confirm_query,
        {
            "confirm": "confirmation_handler",
            "execute": "chatbot",
        }
    )
    graph_builder.add_edge("confirmation_handler", END)
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        }
    )
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile(checkpointer=checkpointer)
from dotenv import load_dotenv
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph
import os
import uuid
import pyttsx3
import re
import subprocess
import sys

# Load .env file from the specific location
load_dotenv(r'C:\Users\Chait\Desktop\Cursor(voice)\.env')

MONGODB_URI = "mongodb://localhost:27017"

def speak(text: str):
    """Simple text-to-speech using pyttsx3"""
    if text and text.strip():  # Only speak if there's actual content
        pyttsx3.speak(text)

def extract_speech_content(response_text: str) -> str:
    """Extract only the SPEECH_OUTPUT part from the response"""
    # Look for SPEECH_OUTPUT: pattern
    speech_match = re.search(r'SPEECH_OUTPUT:\s*(.+?)(?=\n\n|DETAILED_RESPONSE:|$)', response_text, re.DOTALL)
    
    if speech_match:
        speech_content = speech_match.group(1).strip()
        # Clean up any remaining formatting
        speech_content = re.sub(r'\n+', ' ', speech_content)
        return speech_content
    
    return ""  # Return empty string if no speech content found

def get_user_confirmation(use_voice=True):
    """Get user confirmation via voice or text"""
    if use_voice:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please confirm (say 'yes' or 'no'):")
            audio = r.listen(source)
            try:
                response = r.recognize_google(audio).lower()
                return 'yes' in response or 'correct' in response or 'right' in response or 'proceed' in response
            except:
                return False
    else:
        response = input("Please confirm (type 'yes' or 'no'): ").lower()
        return 'yes' in response or 'correct' in response or 'right' in response or 'proceed' in response

def get_user_input(use_voice=True):
    """Get user input via voice or text"""
    if use_voice:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
            try:
                print("Processing audio...")
                sst = r.recognize_google(audio)
                print("You Said:", sst)
                return sst
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Error with speech recognition: {e}")
                return None
    else:
        return input("Type your request: ")

def choose_input_mode():
    """Let user choose between voice and text input"""
    print("\n" + "="*50)
    print("🎤 VOICE-CONTROLLED CODING ASSISTANT")
    print("="*50)
    print("Choose your input mode:")
    print("1. Voice Input (speak your commands)")
    print("2. Text Input (type your commands)")
    print("3. Mixed Mode (ask each time)")
    print("="*50)
    
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == '1':
            return 'voice'
        elif choice == '2':
            return 'text'
        elif choice == '3':
            return 'mixed'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def choose_speech_output():
    """Let user choose if they want voice output"""
    print("\nDo you want voice output (AI speaking responses)?")
    print("1. Yes (AI will speak)")
    print("2. No (text only)")
    
    while True:
        choice = input("Enter your choice (1/2): ").strip()
        if choice == '1':
            return True
        elif choice == '2':
            return False
        else:
            print("Invalid choice. Please enter 1 or 2.")

def get_input_mode_for_turn(current_mode):
    """For mixed mode, ask user for input preference each turn"""
    if current_mode != 'mixed':
        return current_mode
    
    print("\nChoose input for this request:")
    print("1. Voice")
    print("2. Text")
    
    while True:
        choice = input("Enter choice (1/2): ").strip()
        if choice == '1':
            return 'voice'
        elif choice == '2':
            return 'text'
        else:
            print("Invalid choice. Please enter 1 or 2.")

def execute_command(command, shell=True):
    """Execute a command and return the result"""
    try:
        if shell:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        else:
            result = subprocess.run(command, capture_output=True, text=True, cwd=os.getcwd())
        
        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'returncode': -1
        }

def process_and_execute_commands(response_text):
    """Extract and execute commands from the response"""
    # Look for PowerShell commands in the response
    command_patterns = [
        r'```powershell\s*\n(.*?)\n```',
        r'```bash\s*\n(.*?)\n```',
        r'```cmd\s*\n(.*?)\n```'
    ]
    
    executed_commands = []
    
    for pattern in command_patterns:
        commands = re.findall(pattern, response_text, re.DOTALL)
        for command_block in commands:
            # Split command block into individual commands
            individual_commands = command_block.strip().split('\n')
            
            for cmd in individual_commands:
                cmd = cmd.strip()
                if cmd and not cmd.startswith('#') and not cmd.startswith('//'):
                    print(f"Executing: {cmd}")
                    result = execute_command(cmd)
                    executed_commands.append({
                        'command': cmd,
                        'result': result
                    })
                    
                    if result['success']:
                        if result['output']:
                            print(f"Output: {result['output']}")
                    else:
                        print(f"Error: {result['error']}")
    
    return executed_commands

def main():
    # Choose input and output modes
    input_mode = choose_input_mode()
    enable_speech_output = choose_speech_output()
    
    # Initial greeting
    greeting = "I am a helpful AI assistant which can take input in voice or text and make interesting programs for you"
    print(f"\n{greeting}")
    if enable_speech_output:
        speak(greeting)
    
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        
        # Initialize speech recognizer only if needed
        if input_mode in ['voice', 'mixed']:
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    print("Adjusting for ambient noise...")
                    r.adjust_for_ambient_noise(source)
                    r.pause_threshold = 2
                    print("Microphone ready!")
            except Exception as e:
                print(f"Microphone setup failed: {e}")
                print("Falling back to text input only...")
                input_mode = 'text'

        while True:
            try:
                # Determine input method for this turn
                current_input_mode = get_input_mode_for_turn(input_mode)
                use_voice = (current_input_mode == 'voice')
                
                # Get user input
                user_input = get_user_input(use_voice)
                
                if user_input is None:
                    print("No input received. Please try again.")
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    if enable_speech_output:
                        speak("Goodbye!")
                    break
                
                # Use a unique thread ID for each conversation
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                
                # Check if this is a run command for existing code
                if any(word in user_input.lower() for word in ['run', 'execute', 'start']) and len(user_input.split()) <= 3:
                    # Direct run command - execute immediately
                    final_response = None
                    for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "is_simple_query": True, "query_confirmed": True}, config, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            final_response = event["messages"][-1]
                            final_response.pretty_print()
                    
                    if final_response:
                        response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
                        speech_content = extract_speech_content(response_text)
                        if speech_content and enable_speech_output:
                            print(f"\nSpeaking: {speech_content}")
                            speak(speech_content)
                else:
                    # Regular processing flow
                    needs_confirmation = False
                    confirmation_response = None
                    final_response = None
                    
                    # First pass - check if confirmation is needed
                    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            last_message = event["messages"][-1]
                            
                            # Check if this is a confirmation request
                            if hasattr(last_message, 'content') and "Should I proceed" in last_message.content:
                                confirmation_response = last_message
                                needs_confirmation = True
                                break  # Stop processing here
                            else:
                                final_response = last_message
                    
                    if needs_confirmation and confirmation_response:
                        # Handle confirmation
                        confirmation_response.pretty_print()
                        response_text = confirmation_response.content if hasattr(confirmation_response, 'content') else str(confirmation_response)
                        speech_content = extract_speech_content(response_text)
                        
                        if speech_content and enable_speech_output:
                            print(f"\nSpeaking: {speech_content}")
                            speak(speech_content)
                        
                        # Get user confirmation
                        if get_user_confirmation(use_voice):
                            print("User confirmed. Executing task...")
                            if enable_speech_output:
                                speak("Great! I'll start working on that now.")
                            
                            # Execute the task with confirmed query - NEW THREAD ID
                            execution_thread_id = str(uuid.uuid4())
                            execution_config = {"configurable": {"thread_id": execution_thread_id}}
                            
                            final_response = None
                            for event in graph.stream({"messages": [{"role": "user", "content": user_input}], "query_confirmed": True}, execution_config, stream_mode="values"):
                                if "messages" in event and event["messages"]:
                                    final_response = event["messages"][-1]
                                    final_response.pretty_print()
                            
                            if final_response:
                                response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
                                
                                # EXECUTE THE COMMANDS AUTOMATICALLY
                                print("\n" + "="*50)
                                print("EXECUTING COMMANDS AUTOMATICALLY...")
                                print("="*50)
                                executed_commands = process_and_execute_commands(response_text)
                                
                                if executed_commands:
                                    print(f"\nExecuted {len(executed_commands)} commands successfully!")
                                    if enable_speech_output:
                                        speak(f"I've executed {len(executed_commands)} commands for you!")
                                else:
                                    print("No commands found to execute.")
                                
                                speech_content = extract_speech_content(response_text)
                                if speech_content and enable_speech_output:
                                    print(f"\nSpeaking: {speech_content}")
                                    speak(speech_content)
                        else:
                            print("User did not confirm. Please try again with a different request.")
                            if enable_speech_output:
                                speak("Okay, please try again with a different request.")
                    
                    elif final_response:
                        # No confirmation needed, display final response
                        final_response.pretty_print()
                        response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
                        speech_content = extract_speech_content(response_text)
                        if speech_content and enable_speech_output:
                            print(f"\nSpeaking: {speech_content}")
                            speak(speech_content)
                
                print("="*50)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again.")

if __name__ == "__main__":
    main()


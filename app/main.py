from dotenv import load_dotenv
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph
import os
import uuid
import pyttsx3
import re

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

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

def get_user_confirmation():
    """Get user confirmation via voice"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please confirm (say 'yes' or 'no'):")
        audio = r.listen(source)
        try:
            response = r.recognize_google(audio).lower()
            return 'yes' in response or 'correct' in response or 'right' in response or 'proceed' in response
        except:
            return False

def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        
        r = sr.Recognizer()

        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 2

            while True:
                print("Say something!")
                audio = r.listen(source)

                print("Processing audio...")
                sst = r.recognize_google(audio)

                print("You Said:", sst)
                
                # Use a unique thread ID for each conversation
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                
                # Check if this is a run command for existing code
                if any(word in sst.lower() for word in ['run', 'execute', 'start']) and len(sst.split()) <= 3:
                    # Direct run command - execute immediately
                    final_response = None
                    for event in graph.stream({"messages": [{"role": "user", "content": sst}], "is_simple_query": True, "query_confirmed": True}, config, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            final_response = event["messages"][-1]
                            final_response.pretty_print()
                    
                    if final_response:
                        response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
                        speech_content = extract_speech_content(response_text)
                        if speech_content:
                            print(f"\nSpeaking: {speech_content}")
                            speak(speech_content)
                else:
                    # Regular processing flow
                    needs_confirmation = False
                    confirmation_response = None
                    final_response = None
                    
                    for event in graph.stream({"messages": [{"role": "user", "content": sst}]}, config, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            last_message = event["messages"][-1]
                            
                            # Check if this is a confirmation request
                            if hasattr(last_message, 'content') and "Should I proceed" in last_message.content:
                                confirmation_response = last_message
                                needs_confirmation = True
                            else:
                                final_response = last_message
                    
                    if needs_confirmation and confirmation_response:
                        # Handle confirmation
                        confirmation_response.pretty_print()
                        response_text = confirmation_response.content if hasattr(confirmation_response, 'content') else str(confirmation_response)
                        speech_content = extract_speech_content(response_text)
                        
                        if speech_content:
                            print(f"\nSpeaking: {speech_content}")
                            speak(speech_content)
                        
                        # Get user confirmation
                        if get_user_confirmation():
                            print("User confirmed. Executing task...")
                            speak("Great! I'll start working on that now.")
                            
                            # Execute the task with confirmed query
                            for event in graph.stream({"messages": [{"role": "user", "content": sst}], "query_confirmed": True}, config, stream_mode="values"):
                                if "messages" in event and event["messages"]:
                                    final_response = event["messages"][-1]
                                    final_response.pretty_print()
                            
                            if final_response:
                                response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
                                speech_content = extract_speech_content(response_text)
                                if speech_content:
                                    print(f"\nSpeaking: {speech_content}")
                                    speak(speech_content)
                        else:
                            print("User did not confirm. Please try again with a different request.")
                            speak("No problem. Please tell me what you'd like me to do differently.")
                    
                    elif final_response:
                        # Simple query - execute directly
                        final_response.pretty_print()
                        response_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
                        speech_content = extract_speech_content(response_text)
                        if speech_content:
                            print(f"\nSpeaking: {speech_content}")
                            speak(speech_content)
                
                print("="*50)

if __name__ == "__main__":
    # Test text-to-speech
    speak("I am a helpful AI assistant which can take input in voice and make interesting programs for you")
    # Then run the main program
    main()

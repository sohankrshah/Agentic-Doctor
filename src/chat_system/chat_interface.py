from crewai import Crew
from src.agents.crew_agents import chat_agent
from src.tasks.crew_tasks import triage_task
from datetime import datetime
from pathlib import Path
import json

patient_context = {}

def set_patient_info(case_id: str, name: str, age: int):
    """Store patient information for the session and load any existing conversation history"""
    patient_context[case_id] = {
        'name': name,
        'age': age,
        'started_at': datetime.now().isoformat(),
        'conversation_history': []
    }
    
    # Load existing conversation history for this case_id if it exists
    chat_log_path = Path("chat_history.json")
    if chat_log_path.exists():
        try:
            with open(chat_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get('case_id') == case_id:
                            patient_context[case_id]['conversation_history'].append({
                                'user': entry.get('patient_input', ''),
                                'assistant': entry.get('agent_response', ''),
                                'timestamp': entry.get('timestamp', '')
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Note: Could not load previous conversation history: {e}")

def log_chat_entry(case_id: str, user_input: str, agent_response: str):
    """Log chat interactions to file"""
    chat_log_path = Path("chat_history.json")
    
    entry = {
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(),
        "patient_input": user_input,
        "agent_response": agent_response
    }
    
    with open(chat_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')

def handle_ai_chat(user_message: str, case_id: str) -> str:
    """
    Handle chat interaction with Dr. Chen AI agent
    
    Args:
        user_message: The patient's message
        case_id: Unique identifier for this case/session
        
    Returns:
        str: Dr. Chen's response
    """
    try:
        # Get patient context
        patient_info = patient_context.get(case_id, {})
        
        # Build comprehensive context with patient info and FULL conversation history
        context_parts = []
        
        # Add patient info
        if patient_info:
            context_parts.append(f"Patient: {patient_info.get('name', 'Unknown')}, Age: {patient_info.get('age', 'Unknown')}")
        
        # Add COMPLETE conversation history (not just last 3)
        conversation_history = patient_info.get('conversation_history', [])
        if conversation_history:
            context_parts.append("\n=== CONVERSATION HISTORY ===")
            for idx, exchange in enumerate(conversation_history, 1):
                context_parts.append(f"Exchange {idx}:")
                context_parts.append(f"Patient said: {exchange['user']}")
                context_parts.append(f"You (Dr. Chen) responded: {exchange['assistant']}")
                context_parts.append("---")
            context_parts.append("=== END CONVERSATION HISTORY ===\n")
        
        # Combine full context with current message
        if context_parts:
            full_input = "\n".join(context_parts) + f"\n\nCURRENT MESSAGE FROM PATIENT: {user_message}"
        else:
            full_input = user_message
        
        # Create crew with single agent and task for chat
        crew = Crew(
            agents=[chat_agent],
            tasks=[triage_task],
            verbose=False  # Set to True for debugging
        )
        
        # Execute the task
        inputs = {
            "patient_input": full_input
        }
        
        result = crew.kickoff(inputs=inputs)
        
        # Extract response text from result
        if hasattr(result, 'raw'):
            response_text = str(result.raw)
        elif hasattr(result, 'output'):
            response_text = str(result.output)
        else:
            response_text = str(result)
        
        # Clean up the response - remove any tool output artifacts
        response_text = response_text.replace('Predicted class: 0, Confidence: 0.18', '').strip()
        response_text = response_text.replace('{', '').replace('}', '').strip()
        
        # Remove any JSON-like structures from the response
        lines = [line for line in response_text.split('\n') if not line.strip().startswith('"')]
        response_text = '\n'.join(lines).strip()
        
        # Store this exchange in conversation history
        if case_id in patient_context:
            patient_context[case_id]['conversation_history'].append({
                'user': user_message,
                'assistant': response_text,
                'timestamp': datetime.now().isoformat()
            })
        
        # Log the interaction
        log_chat_entry(case_id, user_message, response_text)
        
        return response_text
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error: {str(e)}\n\n"
        error_msg += "This might be due to:\n"
        error_msg += "1. API quota exceeded - Please check your OpenAI billing at https://platform.openai.com/account/billing\n"
        error_msg += "2. Invalid API key - Verify your .env file contains: OPENAI_API_KEY=sk-proj-...\n"
        error_msg += "3. Network issues - Check your internet connection\n\n"
        error_msg += "Please try again or contact support if the issue persists."
        return error_msg

def export_chat_to_text(case_id: str) -> str:
    """Export chat history to a text file"""
    try:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        chat_log_path = Path("chat_history.json")
        
        if not chat_log_path.exists():
            return "No chat history found to export."
        
        patient_info = patient_context.get(case_id, {})
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = reports_dir / f"chat_export_{case_id}_{timestamp}.txt"
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AI MEDICAL ASSISTANT - CHAT TRANSCRIPT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Case ID: {case_id}\n")
            if patient_info:
                f.write(f"Patient Name: {patient_info.get('name', 'Unknown')}\n")
                f.write(f"Age: {patient_info.get('age', 'Unknown')}\n")
                f.write(f"Session Started: {patient_info.get('started_at', 'Unknown')}\n")
            f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            with open(chat_log_path, 'r', encoding='utf-8') as log_file:
                for line in log_file:
                    try:
                        entry = json.loads(line.strip())
                        
                        if entry.get('case_id') != case_id:
                            continue
                        
                        timestamp_str = entry.get('timestamp', 'Unknown time')
                        user_input = entry.get('patient_input', '')
                        agent_response = entry.get('agent_response', '')
                        
                        f.write(f"[{timestamp_str}]\n")
                        f.write(f"PATIENT: {user_input}\n\n")
                        f.write(f"DR. CHEN: {agent_response}\n")
                        f.write("\n" + "-" * 80 + "\n\n")
                    
                    except json.JSONDecodeError:
                        continue
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
            f.write("\nDISCLAIMER: This transcript is for informational purposes only.\n")
            f.write("It does not replace professional medical advice, diagnosis, or treatment.\n")
            f.write("Always consult with a qualified healthcare provider for medical concerns.\n")
        
        return f"✅ Chat transcript exported successfully to: {export_path}"
    
    except Exception as e:
        return f"Error exporting chat: {str(e)}"

def get_chat_summary(case_id: str) -> dict:
    """Get summary statistics for a chat session"""
    try:
        chat_log_path = Path("chat_history.json")
        
        if not chat_log_path.exists():
            return {
                'total_messages': 0,
                'user_messages': 0,
                'agent_messages': 0,
                'session_duration': 'Unknown'
            }
        
        total_messages = 0
        user_messages = 0
        agent_messages = 0
        first_timestamp = None
        last_timestamp = None
        
        with open(chat_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    if entry.get('case_id') != case_id:
                        continue
                    
                    total_messages += 1
                    
                    if entry.get('patient_input'):
                        user_messages += 1
                    if entry.get('agent_response'):
                        agent_messages += 1
                    
                    timestamp = entry.get('timestamp')
                    if timestamp:
                        if first_timestamp is None:
                            first_timestamp = timestamp
                        last_timestamp = timestamp
                
                except json.JSONDecodeError:
                    continue
        
        duration = 'Unknown'
        if first_timestamp and last_timestamp:
            try:
                start = datetime.fromisoformat(first_timestamp)
                end = datetime.fromisoformat(last_timestamp)
                diff = end - start
                duration = f"{diff.seconds // 60} minutes"
            except:
                pass
        
        return {
            'total_messages': total_messages,
            'user_messages': user_messages,
            'agent_messages': agent_messages,
            'session_duration': duration,
            'first_message': first_timestamp,
            'last_message': last_timestamp
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'total_messages': 0
        }
from crewai import Crew
from src.agents.crew_agents import (
    chat_agent,
    lab_agent,
    image_agent,
    research_agent,
    symptom_agent,
    diet_agent,
    wellness_agent,
    followup_agent,
    report_agent,
    vision_agent,
    collab_agent
)
from src.tasks.crew_tasks import (
    triage_task,
    lab_analysis_task,
    image_analysis_task,
    research_task,
    symptom_classification_task,
    diet_task,
    wellness_task,
    followup_task,
    report_task,
    vision_task,
    collab_task
)
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

# 🧠 Full Diagnostic Pipeline
def run_diagnostic_pipeline(patient_input: str = None, image_path: str = None, lab_report_path: str = None):
    inputs = {
        "patient_input": patient_input or "General checkup",
        "image_path": image_path if image_path and Path(image_path).exists() else "No image provided",
        "lab_report_path": lab_report_path if lab_report_path and Path(lab_report_path).exists() else "No lab report provided"
    }

    crew = Crew(
        agents=[
            chat_agent, lab_agent, image_agent, research_agent,
            symptom_agent, diet_agent, wellness_agent, followup_agent,
            report_agent, vision_agent, collab_agent
        ],
        tasks=[
            triage_task, lab_analysis_task, image_analysis_task,
            research_task, symptom_classification_task, diet_task,
            wellness_task, followup_task, report_task, vision_task,
            collab_task
        ],
        verbose=True
    )

    try:
        result = crew.kickoff(inputs=inputs)

        # Optional: Save result to JSON
        log = {
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "result": result
        }
        with open("diagnostic_logs.json", "a") as f:
            f.write(json.dumps(log) + "\n")

        return result
    except Exception as e:
        return f"❌ Error executing diagnostic pipeline: {str(e)}"

# 🧪 Run Single Task
def run_single_task(task_type: str, **kwargs):
    task_map = {
        'triage': (chat_agent, triage_task),
        'lab': (lab_agent, lab_analysis_task),
        'image': (image_agent, image_analysis_task),
        'research': (research_agent, research_task),
        'symptom': (symptom_agent, symptom_classification_task),
        'diet': (diet_agent, diet_task),
        'wellness': (wellness_agent, wellness_task),
        'followup': (followup_agent, followup_task),
        'report': (report_agent, report_task),
        'vision': (vision_agent, vision_task),
        'collab': (collab_agent, collab_task)
    }

    if task_type not in task_map:
        return f"❌ Error: Unknown task type '{task_type}'"

    agent, task = task_map[task_type]

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )

    try:
        result = crew.kickoff(inputs=kwargs)
        return result
    except Exception as e:
        return f"❌ Error executing {task_type} task: {str(e)}"



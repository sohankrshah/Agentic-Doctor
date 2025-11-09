from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# Global case log for collaboration
case_log = {}


class GeneratePDFInput(BaseModel):
    """Input schema for GeneratePDFTool."""
    findings: str = Field(..., description="Main diagnostic findings text")
    citations: str = Field(default="", description="Optional citations or references")


class GeneratePDFTool(BaseTool):
    name: str = "Generate PDF Report"
    description: str = (
        "Creates a structured PDF report with findings and optional references. "
        "Suitable for clinical documentation and patient sharing."
    )
    args_schema: Type[BaseModel] = GeneratePDFInput

    def _run(self, findings: str, citations: str = "") -> str:
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", style="B", size=16)
            pdf.cell(0, 10, "Medical Diagnostic Report", ln=True, align="C")
            pdf.ln(5)
            
            # Date
            pdf.set_font("Arial", size=10)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.ln(5)
            
            # Findings
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 10, "Findings:", ln=True)
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 7, findings)
            pdf.ln(5)

            # Citations
            if citations and citations.strip():
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(0, 10, "References:", ln=True)
                pdf.set_font("Arial", size=10)
                
                # Handle both comma-separated and newline-separated citations
                citation_list = citations.split('\n') if '\n' in citations else citations.split(',')
                for citation in citation_list:
                    if citation.strip():
                        pdf.multi_cell(0, 6, f"• {citation.strip()}")

            # Ensure reports directory exists
            Path("reports").mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f"reports/diagnostic_report_{timestamp}.pdf"
            pdf.output(path)
            
            return f"PDF report generated successfully at: {path}"
            
        except Exception as e:
            return f"Error generating PDF: {str(e)}"


class GenerateXAIHeatmapInput(BaseModel):
    """Input schema for GenerateXAIHeatmapTool."""
    image_path: str = Field(..., description="Path to the medical image")


class GenerateXAIHeatmapTool(BaseTool):
    name: str = "Generate XAI Heatmap"
    description: str = (
        "Overlays a synthetic heatmap on the medical image for explainability. "
        "This is a placeholder implementation - replace with actual XAI model in production."
    )
    args_schema: Type[BaseModel] = GenerateXAIHeatmapInput

    def _run(self, image_path: str) -> str:
        try:
            import cv2
            
            # Read the image
            image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image_data is None:
                return f"Error: Could not read image at {image_path}"
            
            # Create synthetic heatmap (replace with actual XAI model)
            heatmap = np.random.rand(*image_data.shape) * 0.5
            
            # Create visualization
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image_data, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(image_data, cmap='gray')
            plt.imshow(heatmap, cmap='jet', alpha=0.4)
            plt.title('XAI Heatmap Overlay')
            plt.axis('off')
            
            # Ensure reports directory exists
            Path("reports").mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"reports/xai_heatmap_{timestamp}.png"
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return f"XAI heatmap generated successfully at: {output_path}"
            
        except Exception as e:
            return f"Error generating XAI heatmap: {str(e)}"


class ManageCaseInput(BaseModel):
    """Input schema for ManageCaseTool."""
    case_id: str = Field(..., description="Unique identifier for the case")
    doctor_name: str = Field(..., description="Name of the doctor adding the message")
    message: str = Field(..., description="Message content")


class ManageCaseTool(BaseTool):
    name: str = "Manage Case"
    description: str = (
        "Appends a message to a shared case log for multi-doctor collaboration. "
        "Returns the full case log for the specified case."
    )
    args_schema: Type[BaseModel] = ManageCaseInput

    def _run(self, case_id: str, doctor_name: str, message: str) -> str:
        try:
            if case_id not in case_log:
                case_log[case_id] = []
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            entry = f"[{timestamp}] Dr. {doctor_name}: {message}"
            case_log[case_id].append(entry)
            
            return "\n".join(case_log[case_id])
            
        except Exception as e:
            return f"Error managing case: {str(e)}"


# Create tool instances
generate_pdf = GeneratePDFTool()
generate_xai_heatmap = GenerateXAIHeatmapTool()
manage_case = ManageCaseTool()
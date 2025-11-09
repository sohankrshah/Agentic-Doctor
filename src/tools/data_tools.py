from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from pathlib import Path
from PIL import Image
import pytesseract
import pdfplumber
import cv2
import pydicom
import nibabel as nib
from Bio import Entrez
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv

load_dotenv()
Entrez.email = os.getenv("PUBMED_EMAIL", "your_email@example.com")

# ---------------------- LAB REPORT TOOL ----------------------

class ExtractLabTextInput(BaseModel):
    file_path: str = Field(..., description="Path to the lab report file")

class ExtractLabTextTool(BaseTool):
    name: str = "Extract Lab Text"
    description: str = "Extracts text from lab reports in PDF, TXT, or image formats using OCR."
    args_schema: Type[BaseModel] = ExtractLabTextInput

    def _run(self, file_path: str) -> str:
        try:
            path = Path(file_path)
            ext = path.suffix.lower()

            if ext == ".pdf":
                with pdfplumber.open(str(path)) as pdf:
                    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                    return text or "No text found in PDF"
            elif ext == ".txt":
                return path.read_text(encoding='utf-8')
            elif ext in [".png", ".jpg", ".jpeg"]:
                image = Image.open(str(path))
                text = pytesseract.image_to_string(image)
                return text or "No text detected in image"
            else:
                return f"Unsupported format: {ext}"
        except Exception as e:
            return f"Error extracting lab text: {str(e)}"

# ---------------------- MEDICAL IMAGE TOOL ----------------------

class ParseMedicalImageInput(BaseModel):
    file_path: str = Field(..., description="Path to the medical image file")

class ParseMedicalImageTool(BaseTool):
    name: str = "Parse Medical Image"
    description: str = "Parses DICOM (.dcm), NIfTI (.nii), or standard images (.png/.jpg)."
    args_schema: Type[BaseModel] = ParseMedicalImageInput

    def _run(self, file_path: str) -> str:
        try:
            path = Path(file_path)
            ext = path.suffix.lower()

            if ext == ".dcm":
                ds = pydicom.dcmread(str(path))
                pixel_array = ds.pixel_array
                return f"DICOM loaded. Shape: {pixel_array.shape}, Type: {pixel_array.dtype}, Patient ID: {getattr(ds, 'PatientID', 'N/A')}"
            elif ext == ".nii":
                img = nib.load(str(path))
                data = img.get_fdata()
                return f"NIfTI loaded. Shape: {data.shape}, Type: {data.dtype}, Affine: {img.affine.shape}"
            elif ext in [".png", ".jpg", ".jpeg"]:
                img_array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    return "Error reading image"
                return f"Image loaded. Shape: {img_array.shape}, Mean Intensity: {img_array.mean():.2f}"
            else:
                return f"Unsupported format: {ext}"
        except Exception as e:
            return f"Error parsing image: {str(e)}"

# ---------------------- PUBMED SEARCH TOOL ----------------------

class SearchPubMedInput(BaseModel):
    topic: str = Field(..., description="Medical topic to search for")
    max_results: int = Field(default=5, description="Max number of results")

class SearchPubMedTool(BaseTool):
    name: str = "Search PubMed"
    description: str = "Searches PubMed for recent studies on a medical topic."
    args_schema: Type[BaseModel] = SearchPubMedInput

    def _run(self, topic: str, max_results: int = 5) -> str:
        try:
            from datetime import datetime
            year = datetime.now().year
            handle = Entrez.esearch(db="pubmed", term=f"{topic} AND {year}[PDAT]", retmax=max_results)
            record = Entrez.read(handle)
            handle.close()

            ids = record["IdList"]
            if not ids:
                return f"No recent studies found for '{topic}'"

            summaries = []
            for i, pubmed_id in enumerate(ids, 1):
                summary_handle = Entrez.esummary(db="pubmed", id=pubmed_id)
                summary = Entrez.read(summary_handle)[0]
                summary_handle.close()

                title = summary.get("Title", "No title")
                authors = ", ".join(summary.get("AuthorList", [])[:3]) or "Unknown authors"
                pub_date = summary.get("PubDate", "Unknown date")

                summaries.append(f"{i}. {title}\n   Authors: {authors}\n   Date: {pub_date}\n   PMID: {pubmed_id}")

            return "\n\n".join(summaries)
        except Exception as e:
            return f"Error searching PubMed: {str(e)}"

# ---------------------- BIOGPT TOOL ----------------------

class BioGPTInput(BaseModel):
    question: str = Field(..., description="Biomedical question")

class BioGPTTool(BaseTool):
    name: str = "BioGPT Biomedical Q&A"
    description: str = "Answers biomedical questions using BioGPT."
    args_schema: Type[BaseModel] = BioGPTInput

    def _run(self, question: str) -> str:
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
            model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
            inputs = tokenizer(question, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=200)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error using BioGPT: {str(e)}"

# ---------------------- CLINICALBERT TOOL ----------------------

class ClinicalBERTInput(BaseModel):
    text: str = Field(..., description="Patient symptom description")

class ClinicalBERTTool(BaseTool):
    name: str = "ClinicalBERT Symptom Classifier"
    description: str = "Classifies symptoms and urgency using ClinicalBERT."
    args_schema: Type[BaseModel] = ClinicalBERTInput

    def _run(self, text: str) -> str:
        try:
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            logits = outputs.logits.detach().numpy()[0]
            predicted = logits.argmax()
            return f"Predicted class: {predicted}, Confidence: {logits[predicted]:.2f}"
        except Exception as e:
            return f"Error using ClinicalBERT: {str(e)}"

# ---------------------- TOOL INSTANCES ----------------------

extract_lab_text = ExtractLabTextTool()
parse_medical_image = ParseMedicalImageTool()
search_pubmed = SearchPubMedTool()
bio_gpt = BioGPTTool()
clinical_bert = ClinicalBERTTool()
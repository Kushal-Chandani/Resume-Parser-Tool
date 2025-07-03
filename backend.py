from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
from docx import Document
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import json
from pdf2image import convert_from_bytes
import pytesseract
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Pydantic model for structured resume data
class ResumeData(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone_number: str | None = None
    companies: list[str] | None = None
    skills: list[str] | None = None
    goals: str | None = None
    education: list[str] | None = None
    summary: str | None = None

# Function to extract text from PDF (with OCR fallback)
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        with pdfplumber.open(file.file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    # Fallback to OCR
                    file.file.seek(0)
                    images = convert_from_bytes(file.file.read())
                    for image in images:
                        text += pytesseract.image_to_string(image) + "\n"
        logger.info(f"Extracted text from PDF: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

# Function to extract text from DOCX
def extract_text_from_docx(file: UploadFile) -> str:
    try:
        doc = Document(file.file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        logger.info(f"Extracted text from DOCX: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading DOCX: {str(e)}")

# Function to clean extracted text
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to validate if the document is a resume
def is_resume(text: str) -> bool:
    # Keyword-based check
    resume_keywords = [
        'resume', 'cv', 'curriculum vitae', 'experience', 'education', 'skills',
        'work history', 'employment', 'projects', 'certifications', 'objective',
        'summary', 'internship', 'degree', 'bachelor', 'master', 'phd'
    ]
    keyword_pattern = r'\b(' + '|'.join(resume_keywords) + r')\b'
    has_keywords = len(re.findall(keyword_pattern, text.lower())) >= 3
    
    # Structural check
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    job_title_pattern = r'\b(Engineer|Developer|Analyst|Manager|Consultant|Intern|Specialist)\b'
    
    has_structure = (
        re.search(email_pattern, text) or
        re.search(phone_pattern, text) or
        re.search(job_title_pattern, text)
    )
    
    # Gemini-based validation
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Determine if the following text is from a resume or CV. A resume typically contains personal information (name, email, phone), education, work experience, skills, and possibly objectives or a summary. Return a JSON object with a single key 'is_resume' set to true or false.

        Text:
        {text[:1000]}
        """
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text)
        gemini_result = result.get('is_resume', False)
    except Exception as e:
        logger.error(f"Gemini validation failed: {str(e)}")
        gemini_result = False
    
    is_resume_doc = (has_keywords and has_structure) or gemini_result
    logger.info(f"Resume validation: keywords={has_keywords}, structure={has_structure}, gemini={gemini_result}, result={is_resume_doc}")
    return is_resume_doc

# Regex-based fallback extraction
def regex_fallback_extraction(text: str) -> dict:
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    name_pattern = r'([A-Z][a-z]+)\s+([A-Z][a-z]+)'
    skills_keywords = r'\b(Python|Java|SQL|Communication|Leadership|Teamwork|AWS|Docker|Flutter|TensorFlow|PyTorch|Scikit-learn|NumPy|Pandas|Matplotlib|Hugging Face|Vertex AI|Google Cloud|Microsoft Azure|Kubernetes|Power BI|Figma|Agile|[A-Z][a-z]+)\b'
    company_pattern = r'\b([A-Z][a-zA-Z\s]+)\s+(Inc|LLC|Corp|Corporation|Company|Co\.|Ltd|Technologies|Pakistan|University)\b'
    
    phone_numbers = re.findall(phone_pattern, text)
    valid_phone = None
    for num in phone_numbers:
        digits = re.sub(r'\D', '', num)
        if len(digits) >= 10:
            valid_phone = num
            break
    logger.info(f"Regex extracted phone: {valid_phone}")
    
    return {
        "email": next(iter(re.findall(email_pattern, text)), None),
        "phone_number": valid_phone,
        "first_name": re.match(name_pattern, text).group(1) if re.match(name_pattern, text) else None,
        "last_name": re.match(name_pattern, text).group(2) if re.match(name_pattern, text) else None,
        "skills": re.findall(skills_keywords, text),
        "companies": [m[0] for m in re.findall(company_pattern, text)],
    }

# Function to parse resume using Gemini API
def parse_resume_with_gemini(text: str) -> dict:
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    You are an expert resume parser. Extract and generate the following information from the provided resume text in a structured JSON format:
    - First name (e.g., Kushal)
    - Last name (e.g., Chandani)
    - Email address (e.g., kushal.chandani2002@gmail.com)
    - Phone number (e.g., +92-321-456-7890, ensure complete format with 10+ digits)
    - Companies (list of company names, e.g., ['Prosper Technologies', '10Pearls Pakistan'])
    - Skills (list of technical and soft skills, e.g., ['Python', 'Leadership'])
    - Goals (career objectives, e.g., 'To lead AI-driven projects in innovative companies')
    - Education (list of degrees and institutions, e.g., ['Bachelor of Science | Computer Science, Habib University'])
    - Summary (professional summary, e.g., 'Experienced data scientist with a focus on AI')
    
    If career goals or professional summary are not explicitly mentioned in the text, generate concise, professional, and contextually relevant ones based on the candidate's skills, work experience, and education. Focus on the most relevant skills (e.g., AI, programming, cloud technologies) and experiences (e.g., internships at notable companies). Ensure the generated goals and summary are specific to the candidate's profile and avoid generic or overly verbose content.

    Ensure the output is a valid JSON object. If any field is not found, return null or an empty list, except for goals and summary, which should be generated if missing. For phone numbers, ensure a complete format (e.g., +92-321-456-7890). Do not invent data beyond generating goals and summary. Handle various resume formats accurately.

    Resume text:
    {text}
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        parsed_data = json.loads(response.text)
        logger.info(f"Gemini parsed data: {parsed_data}")
        
        # Fallback to regex if Gemini fails to extract key fields
        regex_data = regex_fallback_extraction(text)
        for key in ['email', 'phone_number', 'first_name', 'last_name', 'skills', 'companies']:
            if not parsed_data.get(key):
                parsed_data[key] = regex_data.get(key)
        
        return parsed_data
    except json.JSONDecodeError:
        logger.error("Failed to parse Gemini JSON response, using regex fallback")
        parsed_data = regex_fallback_extraction(text)
        # Generate goals and summary using Gemini
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            fallback_prompt = f"""
            Based on the following resume details, generate a concise career goal and professional summary:
            - Skills: {', '.join(parsed_data.get('skills', []))}
            - Companies: {', '.join(parsed_data.get('companies', []))}
            - Education: {', '.join(parsed_data.get('education', []))}
            
            Ensure the career goal is aspirational and specific (e.g., focusing on AI or software development). The professional summary should highlight key strengths and experiences, tailored to the candidate’s profile. Return a JSON object with 'goals' and 'summary' keys.

            Example:
            {
                "goals": "To excel as a data scientist, leveraging expertise in Python and TensorFlow to build innovative AI solutions.",
                "summary": "A Computer Science graduate with experience at Prosper Technologies, specializing in AI and agile development."
            }
            """
            response = model.generate_content(
                fallback_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            fallback_data = json.loads(response.text)
            parsed_data['goals'] = fallback_data.get('goals')
            parsed_data['summary'] = fallback_data.get('summary')
        except Exception as e:
            logger.error(f"Failed to generate fallback goals/summary: {str(e)}")
            parsed_data['goals'] = "To leverage technical expertise to contribute to innovative projects in leading organizations."
            parsed_data['summary'] = "A motivated professional with a strong academic background and experience in delivering technical solutions."
        return parsed_data
    except Exception as e:
        logger.error(f"Error parsing with Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing with Gemini: {str(e)}")

@app.post("/parse-resume", response_model=ResumeData)
async def parse_resume(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX file.")
    
    try:
        if file.content_type == "application/pdf":
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_docx(file)
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    text = clean_text(text)
    
    # Validate if the document is a resume
    if not is_resume(text):
        raise HTTPException(status_code=400, detail="The uploaded document does not appear to be a resume or CV. Please upload a valid resume.")
    
    try:
        parsed_data = parse_resume_with_gemini(text)
        return ResumeData(**parsed_data)
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Resume-Parser-Tool

A Python-based application that parses PDF or DOCX resumes to extract key information such as name, email, phone number, companies, skills, education, career goals, and professional summary. The tool uses the Gemini API for intelligent parsing, with a FastAPI backend and Streamlit frontend. It includes resume validation to ensure only valid resumes or CVs are processed and dynamically generates career goals and summaries when not explicitly present.

Features:

1. File Support: Processes PDF and DOCX resumes, with OCR fallback for scanned PDFs.
2. Resume Validation: Checks if the uploaded document is a resume/CV using keyword and structural analysis, enhanced by Gemini API.
3. Intelligent Parsing: Extracts entities (name, email, phone, companies, skills, education) using Gemini and regex fallbacks.
4. Dynamic Content Generation: Generates context-aware career goals and professional summaries using Gemini, tailored to the candidate’s skills and experience.
5. User-Friendly Interface: Streamlit frontend displays extracted data in a clean, organized format.

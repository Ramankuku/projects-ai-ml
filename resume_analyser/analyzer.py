from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import pdfplumber

from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-small")
model = ChatOpenAI(api_key=API_KEY, model='gpt-4o-mini', temperature=0.4)

def extract_data_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as fp:
            for page in fp.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        return text

    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")

    
@tool
def extract_resume(resume_text: str):
    '''Analyse the resume like skills, summary, education, Experience and all'''
    
    
    resume_template = PromptTemplate(
    input_variables=["resume", "job_title"],
    template="""
You are a Resume Information Extractor.

STRICT RULES (MANDATORY):
- Extract ONLY the information that is EXPLICITLY written in the resume.
- DO NOT rephrase, summarize, improve, or infer anything.
- DO NOT add missing skills, soft skills, or descriptions.
- DO NOT combine or regroup information.
- Use the SAME wording as in the resume.
- If a section is NOT PRESENT, do NOT include it in the output.
- Do NOT add explanations, conclusions, or suggestions.

Resume Text:
--------------------
{resume}
--------------------

Extract and output ONLY the following sections
(ONLY if they exist in the resume):

Profile Summary:
- (copy exact lines from resume)

Technical Skills:
- (copy exact skills as listed)

Soft Skills:
- (copy exact skills ONLY if explicitly written)

Work Experience:
- (copy exact job titles, companies, durations, descriptions)

Projects:
- (copy exact project titles and descriptions)

Education:
- (copy exact education details)

Achievements / Certifications:
- (copy exact text ONLY if present)

OUTPUT FORMAT:
- Use plain text
- Do NOT use bullet points unless resume uses them
- Do NOT reorder content
"""
)
    prompt = resume_template.format(
        resume=resume_text,
    )
    response = model.invoke(prompt)

    return response.content




@tool
def resume_analyser_find_gaps(resume_text: str, job_title: str):
    """
    Extracts resume details and identifies skill gaps
    with special handling for fresher vs experienced candidates.
    """

    resume_gap_template = PromptTemplate(
        input_variables=["resume_text", "job_title"],
        template="""
You are a Resume Evaluation AI used in a corporate hiring system.

STRICT RULES:
- Use ONLY information explicitly present in the resume.
- Do NOT rewrite, summarize, or improve resume content.
- Do NOT infer skills, experience, or seniority.
- Clearly separate resume extraction and gap analysis.
- Skill gaps must align with the candidate's experience level.
- If no gap is found, explicitly say "No major gaps identified".

JOB TITLE:
{job_title}

RESUME CONTENT:
------------------
{resume_text}
------------------

STEP 1: DETERMINE EXPERIENCE LEVEL
- Fresher: No full-time work experience mentioned OR only academic projects / internships
- Experienced: One or more full-time roles mentioned
- If unclear, state "Experience level unclear"

STEP 2: EXTRACT RESUME DETAILS (EXACT WORDING)
[RESUME DETAILS - EXACT EXTRACTION]
Profile Summary:
Technical Skills:
Soft Skills:
Work Experience:
Projects:
Education:
Certifications:

STEP 3: DEFINE JOB ROLE EXPECTATIONS
- If Fresher: list 5–7 fundamental skills expected for an entry-level role
- If Experienced: list 5–7 professional/production-level skills expected
- Base expectations on common industry standards for the job title

[JOB ROLE EXPECTATIONS]

STEP 4: SKILL GAP ANALYSIS
- Compare resume skills with job role expectations Based on Experience Level
- List ONLY skills that are required as a Fresher/Experienced but NOT present in the resume
- Do NOT suggest improvements or learning paths 

[SKILL GAP ANALYSIS]

STEP 5: FINAL VERDICT
- Tell whether this resume if fresher/experienced level [WRITE THIS RESUME RELATED TO FRESHER LEVEL/EXPERIENCED LEVEL]
- Resume suitability for this role: High / Medium / Low
- Base verdict on experience level alignment and skill coverage
"""
    )

    prompt = resume_gap_template.format(
        resume_text=resume_text,
        job_title=job_title
    )

    response = model.invoke(prompt)
    return response.content





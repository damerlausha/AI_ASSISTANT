import json
import logging
import os
import re
from importlib import import_module
from urllib import error, request

from django.shortcuts import render
from dotenv import load_dotenv
from google import genai


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
logger = logging.getLogger(__name__)
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
FALLBACK_GEMINI_MODEL = "gemini-flash-latest"

try:
    wikipedia = import_module("wikipedia")
except ImportError:
    wikipedia = None


TOPIC_RESPONSES = {
    "admission": "For college admission, check the eligibility criteria, required documents, application dates, and seat availability. Keep your marksheets, ID proof, transfer certificate, and passport-size photos ready.",
    "course": "Choose a course based on your interests, career goals, and future job opportunities. Compare subjects, fees, placement records, and faculty before selecting a branch or specialization.",
    "fee": "College fees depend on the course, institution, and category. Check the official fee structure, scholarship details, and tuition payment deadlines carefully before admission.",
    "placement": "College placements usually depend on your skills, internships, project work, and interview preparation. Focus on resume building, communication skills, aptitude practice, and coding ability.",
    "exam": "College exams usually include internal assessments, assignments, practicals, and end-semester exams. Regular study, previous papers, and time management help you score better.",
    "attendance": "Attendance matters because many colleges require a minimum percentage for exam eligibility. Attend classes regularly, complete assignments, and stay updated with your academic schedule.",
    "faculty": "Faculty members guide you in academics, projects, and career planning. Talk to your mentor or department head when you need academic or personal support.",
    "campus": "A good campus life includes academics, clubs, events, sports, and student support systems. Take part in activities that build confidence, teamwork, and leadership skills.",
    "library": "The library helps you with textbooks, reference materials, journals, and study spaces. Use it to prepare for exams and complete assignments more effectively.",
    "hostel": "Hostel life helps students stay close to campus and build independence. Make sure you understand rules, mess facilities, safety, and room allotment details.",
    "scholarship": "Scholarships are often available based on merit, financial need, or category. Check college notices, government portals, and scholarship forms early to avoid missing deadlines.",
    "internship": "Internships give practical experience and improve your resume. Try to build project work, learn industry tools, and apply for internships before graduation.",
    "project": "Projects help you understand real-world application of theory. Working on mini-projects, labs, and coding tasks improves confidence and placement chances.",
    "result": "Your result reflects your performance in exams and assignments. If your marks are low, focus on weak subjects, practice more, and ask faculty for guidance.",
    "timetable": "The timetable helps you plan classes, labs, practicals, and study time. Follow it regularly to stay organized and avoid missing important lectures.",
    "degree": "A degree gives you formal academic qualification, while skills and internships improve your job readiness. Balance both for better career growth.",
    "college": "A college is a place for learning, academic growth, social development, and career preparation. Students should focus on studies, discipline, and skill-building.",
}


def contains_any(text, keywords):
    normalized_text = text.lower()
    for keyword in keywords:
        pattern = r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)"
        if re.search(pattern, normalized_text):
            return True
    return False


def fetch_wikipedia_summary(question):
    question_lower = (question or "").strip().lower()
    if not question_lower or wikipedia is None:
        return ""

    search_text = re.sub(
        r"^(what is|what are|who is|who was|where is|when was|tell me about|explain)\s+",
        "",
        question_lower,
    ).strip(" ?")
    if not search_text:
        return ""

    try:
        search = wikipedia.search(search_text, results=3)
        if not search:
            return ""

        summary = wikipedia.summary(search[0], sentences=3, auto_suggest=False)
        clean_summary = re.sub(r"\s+", " ", summary).strip()
        if len(clean_summary) > 500:
            clean_summary = clean_summary[:500].rsplit(" ", 1)[0] + "..."
        return clean_summary
    except Exception:
        return ""


def get_gemini_answer(question):
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        return ""

    prompt = (
        "Answer the user's question in a Wikipedia-style way: start with a direct answer, "
        "then give the most important facts and helpful context. You can answer questions "
        "about history, science, geography, people, technology, education, coding, math, "
        "culture, and MRECW (Malla Reddy College of Engineering for Women). Be accurate, "
        "clear, and neutral. Do not invent facts, citations, statistics, or current details. "
        "For changing information such as news, fees, deadlines, rankings, or contact details, "
        "say that it should be verified with a current official source. If the question is "
        "ambiguous, ask one concise clarification.\n\n"
        f"Question: {question}"
    )

    try:
        client = genai.Client(api_key=api_key)
        model = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
        try:
            response = client.models.generate_content(model=model, contents=prompt)
        except Exception:
            if model == FALLBACK_GEMINI_MODEL:
                raise
            logger.warning("Gemini model %s failed; trying %s", model, FALLBACK_GEMINI_MODEL)
            response = client.models.generate_content(
                model=FALLBACK_GEMINI_MODEL,
                contents=prompt,
            )
        return (response.text or "").strip()
    except Exception:
        logger.exception("Gemini could not answer the question")
        return ""


def build_general_answer(question):
    question_lower = (question or "").strip().lower()

    if not question_lower:
        return "Please ask your question."

    if "usha" in question_lower:
        return "Hello Usha! How can I help you today?"

    if contains_any(question_lower, ["mrecw", "malla reddy college of engineering for women"]):
        return (
            "MRECW is Malla Reddy College of Engineering for Women, an autonomous women's "
            "engineering college in Hyderabad, Telangana. It offers technical education and "
            "student programs for women. For current courses, admissions, fees, placements, "
            "and contact details, verify the information on the official college website."
        )

    greeting_patterns = ["hello", "hi", "hey", "good morning", "good evening"]
    if question_lower in greeting_patterns or contains_any(question_lower, greeting_patterns):
        return "Hello! I can help with academic, career, technical, and general student questions."

    gemini_answer = get_gemini_answer(question)
    if gemini_answer:
        return gemini_answer

    if "artificial intelligence" in question_lower or contains_any(question_lower, ["ai"]):
        return (
            "AI (artificial intelligence) is the field of computer science focused on systems that perform tasks such as learning, reasoning, language understanding, and pattern recognition. "
            "Machine learning is a major part of AI in which systems learn from data instead of following only manually written rules. "
            "Common applications include search, recommendations, fraud detection, healthcare support, education tools, and automation."
        )

    if "machine learning" in question_lower:
        return "Machine learning is a branch of artificial intelligence in which a system learns patterns from examples and uses them to make predictions or decisions."

    wiki_answer = fetch_wikipedia_summary(question)
    if wiki_answer:
        return wiki_answer

    for keyword, answer in TOPIC_RESPONSES.items():
        if keyword in question_lower:
            return answer

    if "python" in question_lower:
        return "Python is a beginner-friendly programming language used for web development, automation, data science, AI, and scripting. Start with variables, loops, functions, and OOP concepts."

    if "java" in question_lower:
        return "Java is a widely used object-oriented programming language used in web apps, Android development, and enterprise systems."

    if "javascript" in question_lower:
        return "JavaScript is used to make websites interactive. It works with HTML and CSS for front-end development and can also run on servers with Node.js."

    if "data science" in question_lower:
        return "Data science combines statistics, programming, and domain knowledge to analyze data and make business or research decisions."

    if "cybersecurity" in question_lower:
        return "Cybersecurity focuses on protecting computers, networks, and data from attacks, hacking, and unauthorized access."

    if "study" in question_lower or "study habits" in question_lower:
        return "To study better, use a fixed timetable, take notes, revise regularly, solve problems, and focus on understanding rather than memorizing."

    if "exam" in question_lower:
        return "To prepare for exams, revise key concepts, practice previous questions, manage time well, and stay calm during the test."

    if "career" in question_lower:
        return "Choose a career based on your interests, skills, and long-term goals. Learn practical skills, build projects, and gain experience through internships."

    if "resume" in question_lower:
        return "A good resume should clearly show your education, skills, projects, certifications, and internships in a simple format."

    if "internship" in question_lower:
        return "An internship helps you gain real-world experience, improve your resume, and understand how industries work before graduation."

    if question_lower.startswith("tell me about "):
        topic = question_lower.replace("tell me about ", "", 1).strip().strip("?")
        return f"{topic.title()} is a broad topic that can be understood by learning the basics, examples, and real-world applications. If you want, I can explain it in a simple student-friendly way."

    if question_lower.startswith("what is "):
        topic = question_lower.replace("what is ", "", 1).strip().strip("?")
        return f"{topic.title()} is a concept or subject that can be studied in detail. For a clear explanation, break it into basics, examples, and real-world use cases."

    if question_lower.startswith("who is "):
        topic = question_lower.replace("who is ", "", 1).strip().strip("?")
        return f"{topic.title()} is a person, concept, or idea that may have a specific role depending on the context. If you share more details, I can explain it more clearly."

    if contains_any(question_lower, ["how to", "how can", "how do"]):
        return "Start by understanding the goal, break it into small steps, practice regularly, and ask for help when needed. Clear steps and consistent effort usually lead to better results."

    if contains_any(question_lower, ["why", "reason"]):
        return "The reason depends on the topic, but in general, understanding the purpose, benefits, and real-life use helps you learn better and make smarter decisions."

    if contains_any(question_lower, ["when", "date", "time"]):
        return "The exact time or date depends on the event, course, or schedule. Always check the official calendar, timetable, or notification for accurate details."

    if contains_any(question_lower, ["where", "location"]):
        return "The best place depends on the purpose. For learning, use trusted websites, books, class notes, and official resources. For physical locations, check the official address or campus information."

    if contains_any(question_lower, ["who", "teacher", "mentor", "person"]):
        return "A teacher, mentor, or guide can support you with learning, decision-making, and career planning. It is helpful to ask questions and get feedback from experienced people."

    return "I can help with many kinds of questions, including technology, academics, careers, study tips, and general student life. Please ask a specific question and I will give a clearer answer."


def home(request):
    answer = ""

    if request.method == "POST":
        question = (request.POST.get("question") or "").strip()
        answer = build_general_answer(question)

    return render(
        request,
        "home.html",
        {"answer": answer},
    )
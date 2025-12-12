import os
import re
import tempfile
import logging
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from werkzeug.utils import secure_filename

import pdfplumber
import PyPDF2
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv

# ---------------------------
# Load environment & logging
# ---------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc'}
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)

# ---------------------------
# Certification keywords (partial sample; expand as needed)
# ---------------------------
CERT_KEYWORDS = [
    # AWS
    "AWS Certified Cloud Practitioner", "AWS Certified Solutions Architect Associate",
    "AWS Certified Solutions Architect Professional", "AWS Certified Developer Associate",
    "AWS Certified DevOps Engineer Professional", "AWS Certified SysOps Administrator Associate",
    "AWS Certified Security Specialty", "AWS Certified Machine Learning Specialty",
    "AWS Certified Data Analytics Specialty", "AWS Certified Advanced Networking Specialty",
    "AWS Certified Database Specialty",
    # Azure
    "Microsoft Certified Azure Fundamentals", "Microsoft Certified Azure Administrator Associate",
    "Microsoft Certified Azure Developer Associate", "Microsoft Certified Azure Solutions Architect Expert",
    "Microsoft Certified Azure DevOps Engineer Expert", "Azure Security Engineer Associate",
    "Azure Data Engineer Associate", "Azure AI Engineer Associate", "Azure Network Engineer Associate",
    # GCP
    "Google Associate Cloud Engineer", "Google Professional Cloud Architect",
    "Google Professional Data Engineer", "Google Professional Cloud Network Engineer",
    "Google Professional Cloud Security Engineer", "Google Professional Machine Learning Engineer",
    # Cisco
    "CCNA", "CCNP", "CCIE", "Cisco Certified Network Associate", "Cisco Certified Network Professional", "Cisco CyberOps Associate" , "Cisco Certified" , "Design Associate (CCDA)", "Cisco Certified Design Professional (CCDP)" , "Cisco Security Certification" , "Cisco Collaboration Certification" , "Cisco Wireless Certification",
    # Red Hat
    "RHCSA", "RHCE", "RHCA", "Red Hat Certified System Administrator", "Red Hat Certified Engineer",
    # VMware
    "VCP", "VCAP", "VCDX", "VMware Certified Professional", "VMware Certified Advanced Professional",
    # Linux / Kubernetes
    "LFCS", "LFCE", "CKA", "CKAD", "CKS", "Kubernetes Certified Administrator",
    # CompTIA
    "CompTIA A+", "CompTIA Network+", "CompTIA Security+", "CompTIA Linux+", "CompTIA Cloud+",
    "CompTIA CySA+", "CompTIA PenTest+", "CompTIA CASP+",
    # ISC2
    "CISSP", "CCSP", "SSCP", "HCISPP", "CISSP (Certified Information Systems Security Professional)" , "CCSP (Certified Cloud Security Professional)", "SSCP(Systems Security Certified Practitioner)", "HCISPP (Healthcare Security Certification)", "CAP Certification", "CSSLP Certification",
    # ISACA
    "CISA", "CISM", "CRISC", "CGEIT",
    # EC-Council
    "CEH", "CHFI", "ECSA", "LPT", "OSCP",
    # Offensive Security / GIAC / SANS
    "Offensive Security OSCP", "OSCP", "OSWE", "OSEP", "GIAC GSEC", "GIAC GPEN", "GIAC GCIH",
    # Project & ITIL & Agile
    "PMP", "PRINCE2", "ITIL", "Certified ScrumMaster", "CSM", "SAFe Agilist", "CAPM", "PMP (Project Management Professional)", "CAPM (Certified Associate in Project Management)", "PRINCE2 Foundation", "PRINCE2 Practitioner", "ITIL Foundation", "ITIL Intermediate", "ITIL Practitioner", "Scrum Master", "Scrum Product Owner", "Agile Coach Certification", "SAFe Agilist Certification",
    # Oracle / Salesforce / Others
    "Oracle Certified", "OCI Architect", "Salesforce Administrator", "Salesforce Platform Developer",
    # Add many more as required...
]

# Create lower-case patterns for quick searching (and preserve original mapping)
CERT_PATTERNS = [c.lower() for c in CERT_KEYWORDS]

# ---------------------------
# File utilities
# ---------------------------
def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def secure_temp_save(file_storage) -> str:
    """
    Save an uploaded file_storage-like object to a secure temp file and return the path.
    Works with Streamlit UploadedFile (has getbuffer()), or file-like objects.
    """
    original_name = getattr(file_storage, "filename", None) or getattr(file_storage, "name", None) or "uploaded"
    filename = secure_filename(original_name)
    suffix = os.path.splitext(filename)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    # Streamlit UploadedFile supports getbuffer(); some file-like objects may support .read()
    try:
        data = file_storage.getbuffer()
    except Exception:
        # fallback to read bytes
        file_storage.seek(0)
        data = file_storage.read()

    with open(tmp_path, "wb") as f:
        f.write(data)

    return tmp_path

# ---------------------------
# Resume parsing
# ---------------------------
def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text_parts.append(t)
    except Exception:
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    t = p.extract_text()
                    if t:
                        text_parts.append(t)
        except Exception as e:
            raise RuntimeError(f"PDF parsing failed: {e}")
    return "\n".join(text_parts).strip()

def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        raise RuntimeError(f"DOCX parsing failed: {e}")

def parse_resume(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    else:
        raise ValueError("Unsupported file type")

def extract_emails_from_text(text: str) -> List[str]:
    return EMAIL_REGEX.findall(text) if text else []

# ---------------------------
# Phone extraction
# ---------------------------
PHONE_CLEAN_RE = re.compile(r"[^\d+]")
PHONE_SEARCH_RE = re.compile(
    r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?(\d[\d\s-]{6,14}\d)"
)

def extract_phone_number(text: str) -> str:
    if not text:
        return ""
    matches = PHONE_SEARCH_RE.findall(text)
    for m in matches:
        candidate = "".join(m)
        cleaned = PHONE_CLEAN_RE.sub("", candidate)
        digits = re.sub(r"\D", "", cleaned)
        if 7 <= len(digits) <= 15:
            if cleaned.startswith("+"):
                return "+" + digits
            else:
                return digits
    return ""

# ---------------------------
# TF-IDF scoring (cosine * 100)
# ---------------------------
class TFIDFScorer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = None
        self.tfidf_matrix = None

    def fit_transform(self, job_description: str, resume_texts: List[str]):
        docs = [job_description] + resume_texts
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        return self.tfidf_matrix[0:1], self.tfidf_matrix[1:]

    def score_all(self, jd_vec, resume_vecs) -> List[float]:
        if resume_vecs is None or jd_vec is None:
            return []
        sims = cosine_similarity(resume_vecs, jd_vec).reshape(-1)
        scores = (sims * 100.0).clip(min=0.0, max=100.0)
        return scores.tolist()

    def pairwise_resume_similarities(self, resume_vecs):
        if resume_vecs is None:
            return None
        return cosine_similarity(resume_vecs)

# ---------------------------
# Email service
# ---------------------------
class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.sender_email = os.getenv("SENDER_EMAIL", self.smtp_user)

    def send_shortlist_email(self, to_address, candidate_name, job_description, score, template=None):
        subject = "Congratulations — You have been shortlisted"
        if template:
            body = template.replace("{{candidate_name}}", candidate_name)\
                           .replace("{{score}}", f"{score:.2f}")\
                           .replace("{{job_role}}", job_description)
        else:
            body = f"Dear {candidate_name},\n\nYou are shortlisted.\nScore: {score:.2f}%\n\nRegards,\nHR Team"

        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.sender_email, to_address, msg.as_string())

# ---------------------------
# Reject reason helper
# ---------------------------
def reject_reason_for_code(code: str) -> str:
    reasons = {
        "missing_email": "Invalid or missing Gmail address. Resume does not contain a valid gmail.com email ID.",
        "non_gmail": "Email domain not supported. Only gmail.com email addresses are allowed.",
        "duplicate_email": "This resume was rejected because the same email already exists in the system.",
        "duplicate_content": "Resume content matches an existing resume with very high similarity. Duplicate content detected.",
        "unsupported_file": "Unsupported file format. Only PDF or DOCX resumes are accepted.",
        "parse_failed": "Unable to extract text from the resume. File is corrupted or not readable.",
        "too_short": "Resume content is too short to evaluate. Insufficient information for screening.",
    }
    return reasons.get(code, "Rejected by screening rules.")

# ---------------------------
# Core Resume Processing (enhanced) - now includes certs_found everywhere
# ---------------------------
def process_resumes(job_description, uploaded_files):
    temp_paths = []
    parsed_items = []
    auto_rejected = []
    seen_emails = set()

    try:
        # Step 1: Extract text + metadata (email, phone) for each upload
        for f in uploaded_files:
            filename = getattr(f, "name", None) or getattr(f, "filename", "uploaded")
            if not allowed_file(filename):
                auto_rejected.append({
                    "name": filename,
                    "email": None,
                    "contact_number": "",
                    "reason_code": "unsupported_file",
                    "reason": reject_reason_for_code("unsupported_file"),
                    "text": "",
                    "certs_found": []
                })
                continue

            tmp = secure_temp_save(f)
            temp_paths.append(tmp)

            try:
                text = parse_resume(tmp)
            except Exception as e:
                auto_rejected.append({
                    "name": filename,
                    "email": None,
                    "contact_number": "",
                    "reason_code": "parse_failed",
                    "reason": reject_reason_for_code("parse_failed"),
                    "text": "",
                    "certs_found": []
                })
                continue

            if not text or len(text.strip().split()) < 10:
                contact_number = extract_phone_number(text)
                auto_rejected.append({
                    "name": filename,
                    "email": None,
                    "contact_number": contact_number,
                    "reason_code": "too_short",
                    "reason": reject_reason_for_code("too_short"),
                    "text": text,
                    "certs_found": []
                })
                continue

            emails = extract_emails_from_text(text)
            email = emails[0] if emails else None
            contact_number = extract_phone_number(text)

            if not email:
                auto_rejected.append({
                    "name": filename,
                    "email": None,
                    "contact_number": contact_number,
                    "reason_code": "missing_email",
                    "reason": reject_reason_for_code("missing_email"),
                    "text": text,
                    "certs_found": []
                })
                continue

            if not email.lower().endswith("@gmail.com"):
                auto_rejected.append({
                    "name": filename,
                    "email": email,
                    "contact_number": contact_number,
                    "reason_code": "non_gmail",
                    "reason": reject_reason_for_code("non_gmail"),
                    "text": text,
                    "certs_found": []
                })
                continue

            if email.lower() in seen_emails:
                auto_rejected.append({
                    "name": filename,
                    "email": email,
                    "contact_number": contact_number,
                    "reason_code": "duplicate_email",
                    "reason": reject_reason_for_code("duplicate_email"),
                    "text": text,
                    "certs_found": []
                })
                continue

            seen_emails.add(email.lower())

            # Precompute certs_found for each parsed item (so it's present even before scoring)
            text_lower = (text or "").lower()
            found_certs = []
            for pat in CERT_PATTERNS:
                # Use simple substring detection (consistent with prior logic)
                if pat in text_lower:
                    found_certs.append(pat)

            parsed_items.append({
                "name": filename,
                "email": email,
                "contact_number": contact_number,
                "text": text,
                "certs_found": found_certs  # <-- store detected certs here
            })

        # Step 2: If any parsed_items exist, compute TF-IDF similarity vs job desc
        ranked = []
        if parsed_items:
            scorer = TFIDFScorer()
            jd_vec, resume_vecs = scorer.fit_transform(job_description, [p["text"] for p in parsed_items])
            base_scores = scorer.score_all(jd_vec, resume_vecs)
            pairwise = scorer.pairwise_resume_similarities(resume_vecs)

            for i, p in enumerate(parsed_items):
                # certs_found already computed above; keep it
                found_certs = p.get("certs_found", []) or []
                cert_count = len(found_certs)
                cert_bonus = min(cert_count * 10, 50)

                # Duplicate content detection
                dup_flag = False
                max_sim_other = 0.0
                if pairwise is not None:
                    row = pairwise[i]
                    for j, val in enumerate(row):
                        if i == j:
                            continue
                        pct = val * 100.0
                        if pct > max_sim_other:
                            max_sim_other = pct
                    if max_sim_other >= 90.0:
                        dup_flag = True

                base_score = round(base_scores[i], 2)
                final_score = round(min(base_score + cert_bonus, 100.0), 2)

                reason = ""
                reason_code = ""
                if dup_flag:
                    reason_code = "duplicate_content"
                    reason = reject_reason_for_code("duplicate_content")

                ranked.append({
                    "name": p["name"],
                    "email": p["email"],
                    "contact_number": p["contact_number"] or "",
                    "text": p["text"],
                    "base_score": base_score,
                    "certs_found": found_certs,            # <-- included in ranked
                    "cert_count": cert_count,
                    "cert_bonus": cert_bonus,
                    "final_score": final_score,
                    "duplicate_max_similarity": round(max_sim_other, 2),
                    "is_duplicate_content": dup_flag,
                    "reason_code": reason_code,
                    "reason": reason
                })

            ranked.sort(key=lambda x: x["final_score"], reverse=True)

        return ranked, auto_rejected

    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="VisionDesk", layout="wide")
st.title("VisionDesk — TF-IDF Resume Screening")

# ---------------------------
# Session State
# ---------------------------
if "ranked" not in st.session_state:
    st.session_state.ranked = None
if "auto_rejected" not in st.session_state:
    st.session_state.auto_rejected = []
if "selected_emails" not in st.session_state:
    st.session_state.selected_emails = set()
if "send_logs" not in st.session_state:
    st.session_state.send_logs = []
if "job_description" not in st.session_state:
    st.session_state.job_description = ""

# ---------------------------
# UI Inputs
# ---------------------------
job_description = st.text_area("Job Description")
uploaded_files = st.file_uploader("Upload Resumes", accept_multiple_files=True)

# ---------------------------
# Rank Action
# ---------------------------
if st.button("Rank Resumes"):
    st.session_state.selected_emails = set()
    ranked, rejected = process_resumes(job_description, uploaded_files)
    st.session_state.ranked = ranked
    st.session_state.auto_rejected = rejected
    st.session_state.job_description = job_description

# ---------------------------
# Display Ranked & Selection
# ---------------------------
if st.session_state.ranked is not None:
    ranked = st.session_state.ranked
    auto_rejected = st.session_state.auto_rejected

    st.subheader("Top Candidates")
    for idx, c in enumerate(ranked, 1):
        # Show certs_found in the UI line for quick visibility
        certs_display = f" | certs: {c['certs_found']}" if c.get("certs_found") else ""
        label = f"{idx}. {c['name']} — {c['email']} — {c['final_score']}%{certs_display}"
        if c["is_duplicate_content"]:
            label += "  (Duplicate Content Detected)"
        chk = st.checkbox(label, key=f"sel{idx}")
        if chk:
            st.session_state.selected_emails.add(c["email"])
        elif c["email"] in st.session_state.selected_emails:
            st.session_state.selected_emails.remove(c["email"])

    st.subheader("Auto-Rejected")
    for ar in auto_rejected:
        st.write(f"- **{ar['name']}** — {ar.get('email')} — {ar.get('reason')} — Contact: {ar.get('contact_number','')}")

    st.subheader("Send Email")
    if st.button("Send Shortlist Emails"):
        svc = EmailService()
        logs = []
        for email in st.session_state.selected_emails:
            cand = next(c for c in ranked if c["email"] == email)
            # avoid sending to duplicate-content flagged candidates
            if cand["is_duplicate_content"]:
                logs.append({"email": email, "status": "skipped_duplicate"})
                continue
            try:
                svc.send_shortlist_email(email, cand["name"], job_description, cand["final_score"])
                logs.append({"email": email, "status": "sent"})
            except Exception as e:
                logs.append({"email": email, "status": f"error: {e}"})
        st.session_state.send_logs = logs
        st.json(logs)

    # ---------------------------
    # REPORT TABLE (extended columns) - now includes certs_found / cert_found
    # ---------------------------
    report_rows = []

    # ranked entries
    for r in ranked:
        final_status = "Shortlisted" if r["email"] in st.session_state.selected_emails and not r["is_duplicate_content"] else ("Rejected" if r["is_duplicate_content"] else ("Shortlisted" if r["email"] in st.session_state.selected_emails else "Rejected"))
        reject_reason = r["reason"] if r["is_duplicate_content"] else ""
        report_rows.append({
            "candidate_name": r["name"],
            "email": r["email"],
            "contact_number": r["contact_number"],
            "cert_count": r["cert_count"],
            "cert_bonus": r["cert_bonus"],
            "certs_found": r.get("certs_found", []),           # <-- list retained here
            "similarity_score": r["base_score"],
            "final_score": r["final_score"],
            "duplicate_max_similarity": r["duplicate_max_similarity"],
            "final_status": final_status,
            "Reject_Resume_Reason": reject_reason or ""
        })

    # auto_rejected entries
    for ar in auto_rejected:
        report_rows.append({
            "candidate_name": ar["name"],
            "email": ar.get("email"),
            "contact_number": ar.get("contact_number", ""),
            "cert_count": 0,
            "cert_bonus": 0,
            "certs_found": ar.get("certs_found", []),         # <-- keep empty list for rejected
            "similarity_score": "",
            "final_score": "",
            "duplicate_max_similarity": "",
            "final_status": "Rejected",
            "Reject_Resume_Reason": ar.get("reason", "")
        })

    # Convert to DataFrame. For CSV export, convert list->string for certs_found
    df_for_display = pd.DataFrame(report_rows)
    df_for_export = df_for_display.copy()
    df_for_export["certs_found"] = df_for_export["certs_found"].apply(lambda x: "; ".join(x) if isinstance(x, (list, tuple)) and x else "")

    st.subheader("📋 Shortlist / Rejected Report")
    st.dataframe(df_for_display, use_container_width=True)

    csv_bytes = df_for_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "visiondesk_report.csv", "text/csv")

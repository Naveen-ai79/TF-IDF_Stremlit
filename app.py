# visiondesk_final_tfidf.py
# VisionDesk — TF-IDF Resume Screening (Enhanced) with improved PDF extraction (pdfplumber + PyPDF2 strategies)
# Save and run: streamlit run visiondesk_final_tfidf.py

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

from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------------------
# Config / Tunables (defaults)
# ---------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc'}
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)

# default values (can be adjusted in sidebar)
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "90.0"))
CERT_BONUS_PER = int(os.getenv("CERT_BONUS_PER", "10"))
CERT_BONUS_CAP = int(os.getenv("CERT_BONUS_CAP", "50"))

# ---------------------------
# Certification keywords (extendable)
# Use canonical display names here.
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
    "CCNA", "CCNP", "CCIE", "Cisco Certified Network Associate", "Cisco Certified Network Professional",
    "Cisco CyberOps Associate", "CCDA", "CCDP",
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
    "CISSP", "CCSP", "SSCP", "HCISPP", "CAP", "CSSLP",
    # ISACA
    "CISA", "CISM", "CRISC", "CGEIT",
    # EC-Council
    "CEH", "CHFI", "ECSA", "LPT", "OSCP",
    # Offensive Security / GIAC / SANS
    "Offensive Security OSCP", "OSCP", "OSWE", "OSEP", "GIAC GSEC", "GIAC GPEN", "GIAC GCIH",
    # Project & Agile
    "PMP", "PRINCE2", "ITIL", "Certified ScrumMaster", "CSM", "SAFe Agilist", "CAPM",
    # Oracle / Salesforce / Others
    "Oracle Certified", "OCI Architect", "Salesforce Administrator", "Salesforce Platform Developer",
]

CERT_PATTERNS = [c.lower() for c in CERT_KEYWORDS]

# ---------------------------
# Utilities
# ---------------------------
def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def secure_temp_save(file_storage) -> str:
    original_name = getattr(file_storage, "filename", None) or getattr(file_storage, "name", None) or "uploaded"
    filename = secure_filename(original_name)
    suffix = os.path.splitext(filename)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    data = file_storage.getbuffer()
    with open(tmp_path, "wb") as f:
        f.write(data)
    return tmp_path

# ---------------------------
# Improved PDF extraction (multiple strategies)
# ---------------------------
def _merge_unique_lines(seq_lines: List[str]) -> List[str]:
    """Return unique lines preserving order (strip and ignore empty)."""
    seen = set()
    out = []
    for ln in seq_lines:
        if not ln:
            continue
        s = ln.strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def _words_to_lines(words: List[dict]) -> List[str]:
    """
    Reconstruct approximate lines from pdfplumber.extract_words output.
    Groups words by same 'top' coordinate (rounded) to form lines.
    """
    if not words:
        return []
    # bucket by rounded 'top' coord to form lines
    lines = {}
    for w in words:
        top_key = int(round(w.get('top', 0)))
        lines.setdefault(top_key, []).append((w.get('x0', 0), w.get('text', '')))
    out = []
    for k in sorted(lines.keys()):
        parts = sorted(lines[k], key=lambda x: x[0])
        line = " ".join(p[1] for p in parts).strip()
        if line:
            out.append(line)
    return out

def extract_text_from_pdf(path: str) -> str:
    """
    Multi-strategy PDF text extraction:
    1) pdfplumber page.extract_text()
    2) pdfplumber page.extract_words() -> reconstruct lines
    3) crop page into left/center/right columns and extract from each
    4) fallback to PyPDF2 text extraction
    Merge results, dedupe lines, and return joined text.
    """
    lines_all = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                # Strategy A: page.extract_text()
                try:
                    t = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if t:
                        lines_all.extend([ln.strip() for ln in t.splitlines() if ln.strip()])
                except Exception:
                    pass

                # Strategy B: page.extract_words() -> lines
                try:
                    words = page.extract_words(use_text_flow=True)
                    w_lines = _words_to_lines(words)
                    lines_all.extend(w_lines)
                except Exception:
                    # fallback to extract_words without flow
                    try:
                        words = page.extract_words()
                        w_lines = _words_to_lines(words)
                        lines_all.extend(w_lines)
                    except Exception:
                        pass

                # Strategy C: crop into columns (left/mid/right) to capture multi-column layouts
                try:
                    page_width = page.width
                    thirds = [0.0, page_width * 0.33, page_width * 0.66, page_width]
                    for i in range(3):
                        left = thirds[i]
                        right = thirds[i + 1]
                        try:
                            cropped = page.crop((left, 0, right, page.height))
                            ct = cropped.extract_text(x_tolerance=1, y_tolerance=1)
                            if ct:
                                lines_all.extend([ln.strip() for ln in ct.splitlines() if ln.strip()])
                            # try words for cropped section too
                            cwords = cropped.extract_words(use_text_flow=True)
                            if cwords:
                                lines_all.extend(_words_to_lines(cwords))
                        except Exception:
                            continue
                except Exception:
                    pass

        # dedupe preserving order
        merged = _merge_unique_lines(lines_all)
        return "\n".join(merged).strip()

    except Exception as e:
        # Fallback: try PyPDF2 extraction page-by-page
        logging.warning(f"pdfplumber extraction failed, falling back to PyPDF2: {e}")
        try:
            text_parts = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for p in reader.pages:
                    try:
                        t = p.extract_text()
                        if t:
                            text_parts.extend([ln.strip() for ln in t.splitlines() if ln.strip()])
                    except Exception:
                        continue
            merged = _merge_unique_lines(text_parts)
            return "\n".join(merged).strip()
        except Exception as e2:
            raise RuntimeError(f"PDF parsing failed: {e2}")

# ---------------------------
# DOCX parsing
# ---------------------------
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
PHONE_SEARCH_RE = re.compile(r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?(\d[\d\s-]{6,14}\d)")

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
# TF-IDF scoring and rest of app (unchanged logic from previous version)
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

        if self.smtp_server and self.smtp_user:
            try:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.sendmail(self.sender_email, to_address, msg.as_string())
            except Exception as e:
                logging.error(f"Email send failed: {e}")
        else:
            logging.info("SMTP not configured — skipping actual send (demo mode).")

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
# Main processing pipeline (same as previously delivered)
# ---------------------------
def process_resumes(job_description: str, uploaded_files) -> Tuple[List[Dict], List[Dict]]:
    temp_paths = []
    parsed_items = []
    auto_rejected = []
    seen_emails = set()

    try:
        for f in uploaded_files:
            filename = getattr(f, "name", None) or getattr(f, "filename", "uploaded")
            if not allowed_file(filename):
                auto_rejected.append({
                    "name": filename,
                    "email": None,
                    "contact_number": "",
                    "reason_code": "unsupported_file",
                    "reason": reject_reason_for_code("unsupported_file"),
                    "text": ""
                })
                continue

            tmp = secure_temp_save(f)
            temp_paths.append(tmp)

            try:
                text = parse_resume(tmp)
            except Exception:
                auto_rejected.append({
                    "name": filename,
                    "email": None,
                    "contact_number": "",
                    "reason_code": "parse_failed",
                    "reason": reject_reason_for_code("parse_failed"),
                    "text": ""
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
                    "text": text
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
                    "text": text
                })
                continue

            if not email.lower().endswith("@gmail.com"):
                auto_rejected.append({
                    "name": filename,
                    "email": email,
                    "contact_number": contact_number,
                    "reason_code": "non_gmail",
                    "reason": reject_reason_for_code("non_gmail"),
                    "text": text
                })
                continue

            if email.lower() in seen_emails:
                auto_rejected.append({
                    "name": filename,
                    "email": email,
                    "contact_number": contact_number,
                    "reason_code": "duplicate_email",
                    "reason": reject_reason_for_code("duplicate_email"),
                    "text": text
                })
                continue

            seen_emails.add(email.lower())
            parsed_items.append({
                "name": filename,
                "email": email,
                "contact_number": contact_number,
                "text": text
            })

        ranked = []
        if parsed_items:
            scorer = TFIDFScorer()
            jd_vec, resume_vecs = scorer.fit_transform(job_description or "", [p["text"] for p in parsed_items])
            base_scores = scorer.score_all(jd_vec, resume_vecs)
            pairwise = scorer.pairwise_resume_similarities(resume_vecs)

            # Build preliminary ranked list (order aligns with pairwise indices)
            for i, p in enumerate(parsed_items):
                # --------------------------
                # Precise cert detection + dedupe
                # --------------------------
                text_lower = (p["text"] or "").lower()
                # Normalize whitespace and newlines
                norm_text = re.sub(r'[\r\n]+', ' ', text_lower)
                norm_text = re.sub(r'\s+', ' ', norm_text).strip()

                found_certs_set = set()
                for canonical in CERT_KEYWORDS:
                    pat = canonical.lower()
                    esc = re.escape(pat)
                    regex = r'(?<!\w)' + esc + r'(?!\w)'
                    if re.search(regex, norm_text):
                        found_certs_set.add(canonical)
                found_certs = sorted(found_certs_set)
                cert_count = len(found_certs)
                cert_bonus = min(cert_count * CERT_BONUS_PER, CERT_BONUS_CAP)

                # Duplicate content detection: check max similarity with any other resume (exclude self)
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
                    if max_sim_other >= DUPLICATE_THRESHOLD:
                        dup_flag = True

                base_score = round(base_scores[i], 2)
                final_score = round(min(base_score + cert_bonus, 100.0), 2)

                reason = ""
                reason_code = ""
                if dup_flag:
                    reason_code = "duplicate_content"
                    reason = reject_reason_for_code("duplicate_content")

                ranked.append({
                    # keep index to align with pairwise
                    "idx": i,
                    "name": p["name"],
                    "email": p["email"],
                    "contact_number": p["contact_number"] or "",
                    "text": p["text"],
                    "base_score": base_score,
                    "certs_found": found_certs,
                    "cert_count": cert_count,
                    "cert_bonus": cert_bonus,
                    "final_score": final_score,
                    "duplicate_max_similarity": round(max_sim_other, 2),
                    "is_duplicate_content": dup_flag,
                    "reason_code": reason_code,
                    "reason": reason,
                    "is_keeper": False
                })

            # --- Group duplicates using union-find and pick keeper by (cert_count, final_score, base_score) ---
            n = len(ranked)
            parent = list(range(n))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            if pairwise is not None:
                for i_row in range(n):
                    for j_col in range(i_row + 1, n):
                        sim_pct = pairwise[i_row, j_col] * 100.0
                        if sim_pct >= DUPLICATE_THRESHOLD:
                            union(i_row, j_col)

            groups = {}
            for idx_in_rank in range(n):
                root = find(idx_in_rank)
                groups.setdefault(root, []).append(idx_in_rank)

            # Choose keeper per group, prefer cert_count, then final_score, then base_score
            for root, members in groups.items():
                if len(members) <= 1:
                    m = members[0]
                    ranked[m]["is_keeper"] = True
                    ranked[m]["is_duplicate_content"] = False
                    ranked[m]["reason"] = ""
                    continue

                # find best keeper by key
                best_idx = members[0]
                best_key = (
                    ranked[best_idx].get("cert_count", 0),
                    ranked[best_idx].get("final_score", 0.0),
                    ranked[best_idx].get("base_score", 0.0),
                    -best_idx
                )
                for m in members[1:]:
                    key = (
                        ranked[m].get("cert_count", 0),
                        ranked[m].get("final_score", 0.0),
                        ranked[m].get("base_score", 0.0),
                        -m
                    )
                    if key > best_key:
                        best_idx = m
                        best_key = key

                # mark keeper
                ranked[best_idx]["is_keeper"] = True
                ranked[best_idx]["is_duplicate_content"] = False
                ranked[best_idx]["reason"] = ""

                # mark others as duplicates
                for m in members:
                    if m == best_idx:
                        continue
                    ranked[m]["is_keeper"] = False
                    ranked[m]["is_duplicate_content"] = True
                    ranked[m]["reason_code"] = "duplicate_content"
                    ranked[m]["reason"] = reject_reason_for_code("duplicate_content")

            # Sort ranked by final_score desc for display
            ranked.sort(key=lambda x: x["final_score"], reverse=True)

        return ranked, auto_rejected

    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="VisionDesk", layout="wide")
st.title("VisionDesk — TF-IDF Resume Screening")

# Sidebar controls - update globals when Apply & Save clicked
with st.sidebar:
    st.header("Settings")
    st.write("Tuning parameters (change if needed)")
    dup_thresh = st.number_input("Duplicate similarity threshold (%)", value=DUPLICATE_THRESHOLD, min_value=50.0, max_value=100.0, step=1.0)
    cert_bonus_per = st.number_input("Cert bonus per found cert", value=CERT_BONUS_PER, min_value=1, max_value=100, step=1)
    cert_bonus_cap = st.number_input("Cert bonus cap", value=CERT_BONUS_CAP, min_value=1, max_value=500, step=1)
    st.write("Note: Non-gmail resumes are rejected by policy. Edit code to change this.")
    if st.button("Apply & Save"):
        # update globals so process_resumes uses them
        globals()['DUPLICATE_THRESHOLD'] = float(dup_thresh)
        globals()['CERT_BONUS_PER'] = int(cert_bonus_per)
        globals()['CERT_BONUS_CAP'] = int(cert_bonus_cap)
        st.success("Settings updated for this session.")

# Input area (left) and summary (right)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload & Job Description")
    job_description = st.text_area("Job Description (paste role, responsibilities, must-haves)")
    uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", accept_multiple_files=True)

    if st.button("Rank Resumes"):
        # use current globals
        ranked, rejected = process_resumes(job_description or "", uploaded_files or [])
        # set session state
        st.session_state.ranked = ranked
        st.session_state.auto_rejected = rejected
        st.session_state.job_description = job_description
        # auto-select keeper emails
        keepers = {r["email"] for r in ranked if r.get("is_keeper")}
        st.session_state.selected_emails = set(keepers)

with col2:
    st.subheader("Summary")
    total_uploaded = len(st.session_state.get("ranked", [])) + len(st.session_state.get("auto_rejected", []))
    shortlisted = len([r for r in st.session_state.get("ranked", []) if r.get("is_keeper") or (r.get("final_score") and not r["is_duplicate_content"])])
    duplicates = len([r for r in st.session_state.get("ranked", []) if r["is_duplicate_content"]])
    rejected_count = len(st.session_state.get("auto_rejected", [])) + len([r for r in st.session_state.get("ranked", []) if r["is_duplicate_content"] and not r.get("is_keeper")])
    st.metric("Total processed", total_uploaded)
    st.metric("Shortlisted/Success", shortlisted)
    st.metric("Duplicates flagged", duplicates)
    st.metric("Rejected", rejected_count)

# Display results
if "ranked" in st.session_state and st.session_state.ranked is not None:
    st.subheader("Top Candidates")
    ranked = st.session_state.ranked
    auto_rejected = st.session_state.auto_rejected

    # Compact table
    display_rows = []
    for idx, c in enumerate(ranked, 1):
        status = "Success" if c.get("is_keeper") else ("Duplicate" if c["is_duplicate_content"] else ("Shortlisted" if c["final_score"] else "Rejected"))
        display_rows.append({
            "No": idx,
            "candidate_name": c["name"],
            "email": c["email"],
            "phone": c["contact_number"],
            "cert_count": c["cert_count"],
            "final_score": c["final_score"],
            "status": status,
            "duplicate_pct": c["duplicate_max_similarity"]
        })
    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df, use_container_width=True)

    st.write("---")
    st.write("Select candidates to send shortlist email (duplicates will be skipped automatically).")
    for idx, c in enumerate(ranked, 1):
        label = f"{idx}. {c['name']} — {c['email']} — {c['final_score']}%"
        if c["is_duplicate_content"] and not c.get("is_keeper"):
            label += "  (Duplicate Content Detected)"
        if c.get("is_keeper"):
            label += "  (Selected as keeper)"
        chk = st.checkbox(label, key=f"sel_{idx}", value=(c.get("is_keeper") and True))
        if chk:
            (st.session_state.setdefault("selected_emails", set())).add(c["email"])
        else:
            if c["email"] in st.session_state.get("selected_emails", set()):
                st.session_state["selected_emails"].remove(c["email"])

        # Info expander
        info_list = []
        info_list.append(f"Base similarity: {c['base_score']}%")
        info_list.append(f"Final score: {c['final_score']}%")
        if c.get("cert_count", 0) > 0:
            info_list.append(f"Certifications detected: {c['cert_count']} (bonus +{c['cert_bonus']})")
            certs_display = ", ".join(c.get("certs_found", [])) or "—"
            info_list.append(f"Detected certs: {certs_display}")
        if c["contact_number"]:
            info_list.append(f"Contact: {c['contact_number']}")
        if c["is_duplicate_content"]:
            info_list.append(f"Duplicate match: {c['duplicate_max_similarity']}% (threshold {DUPLICATE_THRESHOLD}%)")
        if c.get("is_keeper"):
            info_list.append("Marked as keeper (best among duplicates by cert_count → final_score → base_score).")

        if info_list:
            with st.expander("ℹ️ Info", expanded=False):
                for line in info_list:
                    st.write(line)

    st.write("---")
    st.subheader("Auto-Rejected")
    if auto_rejected:
        for ar in auto_rejected:
            st.write(f"- **{ar['name']}** — {ar.get('email') or 'N/A'} — {ar.get('reason')} — Contact: {ar.get('contact_number','N/A')}")
    else:
        st.write("No auto-rejected resumes.")

    st.write("---")
    st.subheader("Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Send Shortlist Emails"):
            svc = EmailService()
            logs = []
            for email in st.session_state.get("selected_emails", set()):
                cand = next((c for c in ranked if c["email"] == email), None)
                if cand is None:
                    logs.append({"email": email, "status": "not found"})
                    continue
                # skip duplicates that are not keepers
                if cand["is_duplicate_content"] and not cand.get("is_keeper"):
                    logs.append({"email": email, "status": "skipped - duplicate"})
                    continue
                svc.send_shortlist_email(email, cand["name"], st.session_state.get("job_description", ""), cand["final_score"])
                logs.append({"email": email, "status": "sent"})
            st.success("Emails processed (see logs below).")
            st.json(logs)
    with col_b:
        # Prepare report
        report_rows = []
        for r in ranked:
            # final_status: Success for keeper, otherwise use existing logic
            if r.get("is_keeper"):
                final_status = "Success"
            else:
                final_status = ("Shortlisted" if r["email"] in st.session_state.get("selected_emails", set()) and not r["is_duplicate_content"]
                                else ("Rejected" if r["is_duplicate_content"]
                                      else ("Shortlisted" if r["email"] in st.session_state.get("selected_emails", set()) else "Rejected")))
            # show reason only for duplicates that are not keepers
            reject_reason = r["reason"] if r["is_duplicate_content"] and not r.get("is_keeper") else ""
            report_rows.append({
                "candidate_name": r["name"],
                "email": r["email"],
                "contact_number": r["contact_number"],
                "cert_count": r["cert_count"],
                "certificates_detected": ", ".join(r.get("certs_found", [])) or "",
                "cert_bonus": r["cert_bonus"],
                "similarity_score": r["base_score"],
                "final_score": r["final_score"],
                "duplicate_max_similarity": r["duplicate_max_similarity"],
                "is_keeper": r.get("is_keeper", False),
                "final_status": final_status,
                "Reject_Resume_Reason": reject_reason or ""
            })
        for ar in auto_rejected:
            report_rows.append({
                "candidate_name": ar["name"],
                "email": ar.get("email"),
                "contact_number": ar.get("contact_number", ""),
                "cert_count": 0,
                "certificates_detected": "",
                "cert_bonus": 0,
                "similarity_score": "",
                "final_score": "",
                "duplicate_max_similarity": "",
                "is_keeper": False,
                "final_status": "Rejected",
                "Reject_Resume_Reason": ar.get("reason", "")
            })
        report_df = pd.DataFrame(report_rows)
        st.download_button("Download CSV report", report_df.to_csv(index=False).encode("utf-8"), "visiondesk_report.csv", "text/csv")

    st.subheader("📋 Shortlist / Rejected Report (Preview)")
    st.dataframe(report_df, use_container_width=True)

else:
    st.info("No resumes processed yet. Upload resumes and click 'Rank Resumes' to begin.")

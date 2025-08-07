import os
import re
import pandas as pd
import string
import gender_guesser.detector as gender
import itertools
from datetime import datetime

def extract_form_no(text):
    def is_valid_form_code(code):
        code = code.strip()
        return (
            len(code) >= 2 and
            any(c.isdigit() for c in code) and
            not re.fullmatch(r'[a-zA-Z]{1,2}', code)
        )
    candidates = re.findall(r'Form\s*(?:No[.:]*)?\s*([0-9A-Za-z\-]+)', text, re.IGNORECASE)
    for candidate in candidates:
        if is_valid_form_code(candidate):
            return candidate.strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.fullmatch(r'Form\s*No[.:]*', line.strip(), re.IGNORECASE):
            for j in range(1, 4):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    if is_valid_form_code(next_line):
                        return next_line
    return ''

def extract_institution_and_department(text):
    institution = ''
    department = ''
    lines = text.splitlines()
    header_block = ' '.join(lines[:25])
    if re.search(r'The University of Chicago\s*(University Clinics|Clinics)?', text, re.IGNORECASE):
        institution = 'The University of Chicago Clinics'
    elif re.search(r'The University of Chicag[o]?\s+University Clinics', header_block, re.IGNORECASE):
        institution = 'The University of Chicago Clinics'
    match = re.search(r'(LABORATORY OF [A-Z ]{5,})', text, re.IGNORECASE)
    if match:
        department = match.group(1).title().strip()
    else:
        fallback_block = re.sub(r'[\r\n]+', ' ', text[:1000])
        fallback = re.search(r'(LABORATORY OF(?: [A-Z\-]{2,}){1,5})', fallback_block.upper())
        if fallback:
            department = fallback.group(1).title().strip()
    return institution, department

def extract_patient_name(text):
    import re

    lines = text.splitlines()

    field_label_blacklist = {
        'Pathological Diagnosis', 'Clinical Diagnosis', 'Operation',
        'Attending Physician', 'Clinic or Floor', 'Gross Description',
        'Microscopic Description', 'Hospital', 'Unit No', 'Date',
        'Name Last-First', 'Name First-Last', 'Name Last First', 'Name First Last',
        'Diagnosis', 'Specimen', 'Examination', 'Description', 'Tissue', 'Blocks',
        'Bacteriology', 'Stored', 'Destroyed', 'Museum', 'OPD', 'Surgical Pathology',
        'Name', 'Path. No', 'Path No', 'No. & Street', 'Street', 'City & State', 'Form'
    }

    forbidden_second_parts = {
        'AGE', 'SEX', 'DATE', 'FLOOR', 'PATH', 'DIAGNOSIS',
        'YEAR', 'MALE', 'FEMALE', 'HISTORY'
    }

    false_name_phrases = {
        'FIRST LAST', 'LAST FIRST', 'FIRST-LAST', 'NO H',
        'NIT NO', 'UNIT NO', 'BIRTH DATE', 'SEX MALE',
        'SEX FEMALE', 'BLOOD TYPE', 'NAME LAST', 'NAME FIRST',
        'CITY & STATE', 'STREET', 'STATE', 'NO. & STREET',
        'DR. NAME', 'DOCTOR NAME', 'LAST-FIRST', 'THE', 'FORM', 'OUTSIDE'
    }

    medical_stopwords = {
        'NORMAL', 'YRS', 'PD', 'DO', 'THE', 'DATE', 'FORM',
        'TUBERCULOSIS', 'TONSILS', 'OOPHORITIS', 'INFANT', 'SURGEON',
        'OUTSIDE', 'IST', 'CARCINOMA', 'ULCER', 'GANGRENE'
    }

    def is_field_label(line):
        cleaned = re.sub(r'[^a-zA-Z ]', '', line).strip().lower()
        return any(label.lower() in cleaned for label in field_label_blacklist)

    def is_noise_phrase(name):
        upper = name.upper().strip().rstrip('.')
        if upper in false_name_phrases or upper in medical_stopwords:
            return True
        if 'LAST' in upper and 'FIRST' in upper:
            return True
        if upper == 'OUTSIDE':
            return True
        if re.fullmatch(r'[A-Z]\.?', upper):
            return True
        if re.fullmatch(r'[A-Z]{2,}', upper):
            return True
        if re.search(r'\b(NO|UNIT|STATE|STREET|FORM|DATE|YR|AGE|SEX|PD|DO)\b', upper):
            return True
        return False

    def looks_like_name(name):
        name = name.strip().rstrip(",")
        if not name or len(name) < 3 or is_noise_phrase(name):
            return False
        if name.lower() in {'ulcer', 'carcinoma', 'outside', 'form'}:
            return False
        if re.fullmatch(r'[a-z]+', name):
            return False
        if re.fullmatch(r'[A-Z] ?\d$', name):
            return False
        if name.startswith("Dr.") or name.startswith("Doctor"):
            return False
        if "'s case" in name.lower():
            return False
        parts = re.split(r'[,\s]+', name)
        parts = [p for p in parts if p]
        if not (1 < len(parts) <= 3):
            return False
        if any(p.upper() in forbidden_second_parts for p in parts[1:]):
            return False
        for part in parts:
            if not re.match(r'^[A-Z][a-zA-Z.\'-]+$', part):
                return False
        return True

    def score_name(name):
        if is_noise_phrase(name) or is_field_label(name):
            return -100
        if 'last' in name.lower() and 'first' in name.lower():
            return -100
        score = 0
        if ',' in name:
            score += 2
        if re.fullmatch(r'[A-Z][a-zA-Z]+, ?[A-Z]\.?', name):
            score += 2
        if re.fullmatch(r'[A-Z][a-z]{2,},', name):
            score += 1  # Allow Kruger,
        if re.match(r'(Mr\.|Mrs\.|Miss) ', name):
            score += 3
        return score

    for i, line in enumerate(lines):
        line_clean = line.strip()
        if re.fullmatch(r'Name(?:[- ]?(Last[- ]?First|First[- ]?Last))?[.:]*', line_clean, re.IGNORECASE):
            for j in range(1, 5):
                if i + j < len(lines):
                    candidate = lines[i + j].strip()
                    if looks_like_name(candidate) and not is_field_label(candidate):
                        return candidate
            if i - 1 >= 0:
                candidate = lines[i - 1].strip()
                if looks_like_name(candidate) and not is_field_label(candidate):
                    return candidate
        inline_match = re.match(
            r'^Name(?:[- ]?(Last[- ]?First|First[- ]?Last))?[.:]*\s*(.+)$',
            line_clean, re.IGNORECASE)
        if inline_match:
            candidate = inline_match.group(2).strip()
            if looks_like_name(candidate) and not is_field_label(candidate):
                return candidate

    for keyword in ['Clinic or Floor', 'Attending Physician']:
        for i, line in enumerate(lines):
            if keyword in line:
                for offset in range(1, 4):
                    for direction in [-1, 1]:
                        idx = i + direction * offset
                        if 0 <= idx < len(lines):
                            candidate = lines[idx].strip()
                            if looks_like_name(candidate) and not is_field_label(candidate):
                                return candidate

    name_candidates = set()
    name_candidates.update(re.findall(r'\b[A-Z][a-zA-Z.\'-]+, ?[A-Z][a-zA-Z.\'-]+\b', text))
    name_candidates.update([
        m.group()
        for m in re.finditer(r'\b[A-Z][a-z]+ [A-Z][a-zA-Z.\'-]+\b', text)
        if not is_field_label(m.group()) and not is_noise_phrase(m.group())
    ])
    name_candidates.update(re.findall(r'\b[A-Z][a-zA-Z]+, ?[A-Z]\.?\b', text)) 
    name_candidates.update(re.findall(r'\b[A-Z][a-zA-Z]+,?\b', text))          # Kruger,
    name_candidates.update(re.findall(r'\b(?:Mr\.|Mrs\.|Miss)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b', text))

    scored = [
        (score_name(name), name)
        for name in name_candidates
        if looks_like_name(name)
    ]
    scored.sort(reverse=True)

    for score, name in scored:
        if score > 0:
            return name

    return ''

detector = gender.Detector()

def guess_gender(name):
    if not name or not isinstance(name, str):
        return 'Unknown'

    title_match = re.search(r'\b(Mr|Mrs|Miss|Ms)\b', name, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).lower()
        if title in ['mrs', 'miss', 'ms']:
            return 'Female'
        elif title == 'mr':
            return 'Male'

    name = re.sub(r'\b(Dr|Mr|Mrs|Ms|Miss)\.?\b', '', name, flags=re.IGNORECASE).strip()

    parts = name.replace(',', ' ').split()
    if not parts:
        return 'Unknown'

    first_name = parts[1] if len(parts) >= 2 else parts[0]

    if len(first_name) <= 2 or re.fullmatch(r'[A-Z]\.?', first_name):
        return 'Unknown'

    manual_dict = {
        'Estelito': 'Male',
        'Orma': 'Female',
        'Linna': 'Female',
        'Noel': 'Male'
    }

    if first_name in manual_dict:
        return manual_dict[first_name]

    g = detector.get_gender(first_name)
    if g in ['male', 'mostly_male']:
        return 'Male'
    elif g in ['female', 'mostly_female']:
        return 'Female'
    elif g == 'andy':
        return 'Uncertain'
    else:
        return 'Unknown'

def extract_age(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'Age of Patient' in line:
            if i + 1 < len(lines):
                age_match = re.search(r'\b(\d{1,3})\b', lines[i + 1])
                if age_match:
                    return age_match.group(1)
    match = re.search(r'Age of Patient[.:]?\s*(\d{1,3})', text, re.IGNORECASE)
    return match.group(1) if match else ''

def extract_unit_no(text):
    import re

    lines = text.splitlines()
    blacklisted_terms = {'form', 'path', 'date', 'no', 'name', 'block', 'clinic', 'floor', 'age'}
    suspicious_prefixes = ('00-', '02-', '14-', 'LO-', 'L-', 'C-', 'R-')

    for i, line in enumerate(lines):
        if re.fullmatch(r'\bUnit\s*N[o0]*[.:]*\s*', line.strip(), re.IGNORECASE):
            digit_lines = []
            for j in range(1, 6):
                if i + j >= len(lines):
                    break
                candidate = lines[i + j].strip()
                if not candidate or not re.search(r'\d', candidate):
                    continue
                if re.fullmatch(r'[\d\s]{1,20}', candidate):
                    digit_lines.append(candidate)
                else:
                    break
            if (
                len(digit_lines) >= 2 and
                all(len(re.sub(r'\D', '', x)) <= 4 for x in digit_lines)
            ):
                digits = ''.join(re.findall(r'\d+', ' '.join(digit_lines)))
                if 4 <= len(digits) <= 8:
                    return digits

    for i, line in enumerate(lines):
        if re.fullmatch(r'\bUnit\s*N[o0]*[.:]*\s*', line.strip(), re.IGNORECASE):
            skip_next = 0
            if i + 1 < len(lines) and re.search(r'\bdate\b', lines[i + 1], re.IGNORECASE):
                skip_next = 2
            for j in range(skip_next + 1, 10):
                if i + j >= len(lines):
                    break
                candidate = lines[i + j].strip()

                if re.fullmatch(r'\d{1,2}\s+\d{3,5}\s+\d{3,5}', candidate):
                    return candidate.split()[1]
                if re.fullmatch(r'\d{4,6}\s+\d{3,6}', candidate):
                    return candidate.split()[0]
                if re.fullmatch(r'\d{1,2}\s+\d{4,6}', candidate):
                    return candidate.split()[1]

                if re.fullmatch(r'\d{4,6}', candidate):
                    ctx = " ".join(lines[max(0, i - 3): i + j + 3]).lower()
                    if 'unit' in ctx and not re.search(r'\b812a\b|\bform\b', ctx):
                        return candidate

                if not candidate or candidate.lower() in blacklisted_terms:
                    continue
                if re.search(r'[\[\]&]', candidate):
                    continue

                parts = re.findall(r'\b[\w\-]{3,}\b', candidate)
                for p in parts:
                    if re.fullmatch(r'\d{4,6}', p):
                        return p

    for i, line in enumerate(lines):
        line = line.strip()
        context = " ".join(lines[max(0, i - 2): i + 3]).lower()
        if 'path' in context and re.search(r'\bpath\s*no\b', context) and 'unit' not in context:
            continue
        if re.fullmatch(r'\d{4,6}\s+\d{3,6}', line):
            return line.split()[0]
        if re.fullmatch(r'\d{1,2}\s+\d{3,5}\s+\d{3,5}', line):
            return line.split()[1]
        if re.fullmatch(r'\d{1,2}\s+\d{4,6}', line):
            return line.split()[1]

    for i, line in enumerate(lines):
        if re.fullmatch(r'[A-Z]?\d{4,6}[A-Z]?', line):
            ctx_lines = lines[max(0, i - 3): i + 4]
            ctx = " ".join(ctx_lines).lower()

            if re.fullmatch(r'19[0-4]\d|195\d', line):
                if any("path no" in l.lower() for l in ctx_lines[:4]) or "date" in ctx:
                    continue
                if not any('unit' in l.lower() for l in ctx_lines):
                    continue

            if (
                re.fullmatch(r'[A-Z]?\d{3,4}[A-Z]?', line) and
                re.search(r'form\s+(no\.\s*)?\d+', ctx)
            ):
                continue
            if line.upper() == "812A":
                continue
            if re.fullmatch(r'0\d{5,}', line) and 'unit' not in ctx:
                continue
            if any(key in ctx for key in ('unit', 'clinic', 'path')):
                return line

    inline_patterns = [
        r'\bUnit\s+N[o0][.:]*\s*([0-9A-Za-z\-]{3,})',
        r'\bUnit[.:]*\s*([0-9A-Za-z\-]{3,})',
        r'\bUnit\s*#\s*([0-9A-Za-z\-]{3,})',
    ]
    for pattern in inline_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(1).strip()
            if not re.search(r'\d', candidate):
                continue
            if candidate.upper() == "812A":
                continue
            if any(bad in candidate.lower() for bad in blacklisted_terms):
                continue
            if candidate.startswith(suspicious_prefixes):
                continue
            if re.fullmatch(r'[Ll]\d{1,3}', candidate):
                continue
            if re.fullmatch(r'[SNEWsnew]-\d{1,2}', candidate):
                continue
            if re.fullmatch(r'\d{6,}', candidate) and candidate.startswith('0'):
                continue
            context = text[max(0, match.start() - 80): match.end() + 80].lower()
            if re.search(r'(refer|slide|mismatch|belongs to|another case)', context):
                continue
            return candidate

    return ''

def fill_missing_unit_no(df, name_col='name', unit_col='unit_no'):
    import numpy as np
    df = df.copy()

    df[unit_col] = df[unit_col].replace('', np.nan)

    df['_name_key'] = df[name_col].fillna('').str.replace(r'\s+', '', regex=True).str.lower()

    df['_row_id'] = df.index

    df['_group_id'] = (df['_name_key'] != df['_name_key'].shift()).cumsum()

    df = df.sort_values(by=['_group_id', '_row_id'])

    df[unit_col] = df.groupby('_group_id')[unit_col].ffill()

    df = df.sort_values('_row_id').drop(columns=['_name_key', '_group_id', '_row_id'])

    return df

def extract_doctor_name(text, field_label, allow_upward=False, exclude_names=None, allow_fallback=False):
    
    exclude_names = exclude_names or set()
    lines = text.splitlines()
    field_label_lower = field_label.lower()

    lines = [re.sub(r'\bD[fgqk,.]?\b', 'Dr.', line) for line in lines]
    lines = [re.sub(r'\bDr\.\.', 'Dr.', line) for line in lines]
    lines = [re.sub(r'\bDR\b', 'Dr.', line) for line in lines]
    lines = [re.sub(r'\bDr\s+', 'Dr. ', line) for line in lines]

    blacklist = {
        "stored", "destroyed", "museum", "date", "paraffin", "celloidin", "hospital",
        "clinical diagnosis", "gross", "blocks", "operation", "disposal", "outside",
        "form", "unit", "clinic or floor", "name", "age", "sex", "yrs", "drawing",
        "description", "specimen", "normal", "index", "", "block", "tissue", "bacteriology",
        "hemorrhoids", "appendicitis", "nephrectomy", "polyp", "tumor", "ulcer", "lymphoma",
        "cyst", "fibrosis", "necrosis", "tuberculosis", "carcinoma", "melanoma", "inflammation"
    }

    anatomy_words = {
        'ovary', 'uterus', 'fallopian', 'salpingitis', 'tube', 'mucosa',
        'section', 'serosa', 'crypts', 'lumen', 'precipitate', 'cavity',
        'histocytes', 'fibroblasts', 'granulation', 'cell', 'infiltration',
        'connective', 'biopsy', 'drawing', 'reproductive', 'description'
    }

    fake_doctor_names = {
        'dr. surgeon', 'dr. attending physician', 'dr. pathologist', 'dr. collaborator'
    }

    def looks_like_doctor_name(name):
        name = re.sub(r"^[^\w]*|[^\w]*$", "", name.strip())
        if not name or len(name) < 3:
            return False
        lower = name.lower()
        if lower in blacklist or lower in anatomy_words or name in exclude_names or lower in fake_doctor_names:
            return False
        if name.upper() in {'MC', 'S', 'M', 'OP', 'IST', 'PORT', 'PRES', 'HOSP', 'TUBE'}:
            return False
        if re.fullmatch(r'\d+|\d+[a-z]?|\w+\d+', name):
            return False
        if any(kw in lower for kw in ['university', 'hospital', 'clinic', 'floor', 'unit', 'path', 'pres']):
            return False
        if name.isupper() and not name.startswith("Dr."):
            return False
        return bool(re.match(r'^(Drs?\.?|Dr,)?\s*[A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+)*$', name))

    def extract_multiple_doctors(line):
        results = re.findall(r'\bDr\.?\s+[A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+)?', line)
        return [r.strip(" .,:;") for r in results if looks_like_doctor_name(r)]

    for i, line in enumerate(lines):
        if re.search(r'^Name[.: -]+', line.strip(), re.IGNORECASE):
            patient = re.sub(r'^Name[.: -]+', '', line, flags=re.IGNORECASE).strip(" .:,;")
            if looks_like_doctor_name(patient):
                exclude_names.add(patient)
            for k in [-2, -1, 1, 2]:
                if 0 <= i + k < len(lines):
                    ln = lines[i + k].strip()
                    if looks_like_doctor_name(ln):
                        exclude_names.add(ln)

    for i, line in enumerate(lines):
        line_clean = line.strip()

        if line_clean.lower().strip(" .:") == field_label_lower:
            for j in range(1, 5):
                if i + j < len(lines):
                    candidate = lines[i + j].strip()
                    if not candidate:
                        continue
                    results = extract_multiple_doctors(candidate)
                    if results:
                        return ' and '.join(results)
                    if looks_like_doctor_name(candidate):
                        return candidate

        if field_label_lower in line_clean.lower():
            after = re.split(rf'{re.escape(field_label)}[\s.:;,_-]*', line_clean, flags=re.IGNORECASE)
            if len(after) > 1:
                suffix = after[1].strip(" .,:;")
                results = extract_multiple_doctors(suffix)
                if results:
                    return ' and '.join(results)
                if looks_like_doctor_name(suffix):
                    return suffix

            for j in range(1, 5):
                if i + j < len(lines):
                    candidate = lines[i + j].strip()
                    if not candidate:
                        continue
                    results = extract_multiple_doctors(candidate)
                    if results:
                        return ' and '.join(results)
                    if looks_like_doctor_name(candidate):
                        return candidate

            if allow_upward:
                for j in range(1, 5):
                    if i - j >= 0:
                        candidate = lines[i - j].strip()
                        if candidate.lower() in fake_doctor_names:
                            continue
                        if looks_like_doctor_name(candidate):
                            return candidate

    if allow_fallback:
        all_candidates = []
        skip_keywords = blacklist.union(anatomy_words)

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if field_label_lower in line_lower:
                continue
            if any(skip in line_lower for skip in skip_keywords):
                continue
            if 'dr' not in line_lower:
                continue

            results = extract_multiple_doctors(line_stripped)
            for r in results:
                if r not in exclude_names and looks_like_doctor_name(r):
                    all_candidates.append((i, r))

            possessive_matches = re.findall(r"\bDr\.?\s+([A-Z][a-z]+)'s case", line_stripped)
            for name in possessive_matches:
                full = f"Dr. {name}"
                if full not in exclude_names and looks_like_doctor_name(full):
                    return full

        for i, line in enumerate(lines):
            if line.strip().lower().strip(" .:") == field_label_lower:
                for j in range(1, 5):
                    if i + j < len(lines):
                        candidate = lines[i + j].strip()
                        if not candidate:
                            continue
                        results = extract_multiple_doctors(candidate)
                        if results:
                            return ' and '.join(results)
                        if looks_like_doctor_name(candidate):
                            return candidate

        field_indices = [i for i, line in enumerate(lines) if field_label_lower in line.lower()]
        local_candidates = []
        for i, name in all_candidates:
            if any(abs(i - fidx) <= 5 for fidx in field_indices):
                local_candidates.append(name)

        unique = list(set(local_candidates))
        if len(unique) == 1:
            return unique[0]
        elif len(unique) > 1:
            return ' and '.join(unique)

    return ''

def extract_attending_physician(text):
    return extract_doctor_name(text, field_label="Attending Physician", allow_upward=True, allow_fallback=True)

def extract_surgeon(text):
    attending = extract_attending_physician(text)
    surgeon = extract_doctor_name(
        text,
        field_label="Surgeon",
        allow_upward=True,
        exclude_names={attending} if attending else set(),
        allow_fallback=True
    )
    return surgeon if surgeon and surgeon != attending else ''

def extract_date(text):
    CURRENT_YEAR = datetime.now().year
    MAX_YEAR = CURRENT_YEAR + 2

    lines = text.splitlines()
    line_flags = [False] * len(lines)

    for i, line in enumerate(lines):
        if re.search(r'(unit|path)\s*no', line, re.IGNORECASE):
            for j in range(i, min(len(lines), i + 5)):
                line_flags[j] = True

    def is_plausible_date(parts, source='strict'):
        try:
            m, d, y = map(int, parts)
            if not (1 <= m <= 12 and 1 <= d <= 31):
                return False
            if y >= 100:
                if y > 9999:
                    return False
                if source == 'strict' and (y < 1900 or y > MAX_YEAR):
                    return False
                if source == 'relaxed' and (y < 1800 or y > MAX_YEAR):
                    return False
            else:
                y_full = 2000 + y if y <= (MAX_YEAR % 100) else 1900 + y
                if y_full > MAX_YEAR:
                    return False
            return True
        except:
            return False

    def normalize_date(parts):
        m, d, y = map(int, parts)
        return f"{m}-{d}-{y}"

    def extract_date_from_line(line, source='strict'):
        line = line.strip()
        if re.search(r'(unit|path)\s*no', line, re.IGNORECASE):
            return ''
        if len(re.findall(r'\d+', line)) > 3 and not re.search(r'[-/\.]', line):
            return ''
        match = re.search(r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b', line)
        if match:
            parts = re.split(r'[-/.]', match.group(0))
            if len(parts) == 3 and is_plausible_date(parts, source=source):
                return normalize_date(parts)
        return ''

    def combine_and_validate(parts, source='strict', context_lines=None):
        candidates = []
        for perm in itertools.permutations(parts, 3):
            if is_plausible_date(perm, source=source):
                m, d, y = map(int, perm)
                if context_lines:
                    if context_lines.get("allow_loose_year") is False:
                        if len(str(perm[0])) == 4 and context_lines.get("first_token_line") == "year_line":
                            continue
                candidates.append(perm)
        if candidates:
            best = sorted(candidates, key=lambda x: (x[2], x[0], x[1]))[0]
            return normalize_date(best)
        return ''

    for i, line in enumerate(lines):
        if re.search(r'\bDate\b[:.\-]?', line, re.IGNORECASE):
            tokens_from_lines = []
            skipped = 0
            k = 1
            while len(tokens_from_lines) < 3 and (i + k) < len(lines) and skipped < 2:
                nums = re.findall(r'\d{1,4}', lines[i + k])
                if nums:
                    first = int(nums[0])
                    if 1900 <= first <= MAX_YEAR:
                        skipped += 1
                        k += 1
                        continue
                    tokens_from_lines.append(str(first))
                k += 1
            if len(tokens_from_lines) == 3:
                guess = combine_and_validate(tokens_from_lines, source='relaxed')
                if guess:
                    return guess

            for j in range(-3, 4):
                idx = i + j
                if 0 <= idx < len(lines) and not line_flags[idx]:
                    found = extract_date_from_line(lines[idx], source='relaxed')
                    if found:
                        return found

    date_candidates = []
    for i, line in enumerate(lines):
        if not line_flags[i]:
            found = extract_date_from_line(line, source='strict')
            if found:
                date_candidates.append(found)
    if date_candidates:
        return date_candidates[0]

    for i, line in enumerate(lines):
        if 'date' in line.lower():
            nearby_digits = []
            for j in range(i, min(i + 6, len(lines))):
                nearby_digits += re.findall(r'\d{1,4}', lines[j])
            if len(nearby_digits) >= 3:
                guess = combine_and_validate(nearby_digits[:6], source='strict')
                if guess:
                    return guess

    for i in range(len(lines) - 2):
        if any(line_flags[i + j] for j in range(3)):
            continue
        tokens_3_lines = []
        for j in range(3):
            tokens = re.findall(r'\d{1,4}', lines[i + j])
            tokens_3_lines.append(tokens if tokens else [])
        if all(tokens_3_lines):
            triplet = [tokens_3_lines[0][0], tokens_3_lines[1][0], tokens_3_lines[2][0]]
            context_info = {
                "allow_loose_year": False,
                "first_token_line": "year_line" if len(triplet[0]) == 4 else "other"
            }
            guess = combine_and_validate(triplet, source='relaxed', context_lines=context_info)
            if guess:
                return guess
            for a in tokens_3_lines[0]:
                for b in tokens_3_lines[1]:
                    for c in tokens_3_lines[2]:
                        guess = combine_and_validate([a, b, c], source='relaxed')
                        if guess:
                            return guess

    return ''

def extract_clinic_or_floor(text):
    def looks_like_clinic_value(val):
        val = val.strip().lower()
        if not val:
            return False
        if val in {'outside'}:
            return True
        return bool(re.match(r'^[a-zA-Z]-?\d+$', val))

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'clinic or floor' in line.lower():
            suffix = re.split(r'clinic or floor[:\s\-.]*', line, flags=re.IGNORECASE)
            if len(suffix) > 1:
                value = suffix[1].strip(" .,:;")
                if value and looks_like_clinic_value(value):
                    return value

            for j in range(1, 3):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    if not next_line or next_line.lower().startswith(('date', 'unit no', 'path.', 'attending')):
                        continue
                    if looks_like_clinic_value(next_line):
                        return next_line
    return ''

def extract_clinical_diagnosis(text):
    lines = text.splitlines()
    field_keywords = ['clinical diagnosis', 'clinical data and diagnosis']

    structural_terms = [
        'date', 'attending', 'name', 'unit no', 'clinic or floor', 'path',
        'stored', 'museum', 'bacteriology', 'gross description', 'microscopic',
        'operation', 'form no', 'no. of blocks', 'specimen', 'consult', 'transferred'
    ]

    medical_keywords = [
        'cancer', 'tumor', 'cyst', 'fistula', 'abscess', 'sarcoma',
        'tonsil', 'carcinoma', 'fibroma', 'adenoma', 'mastoiditis',
        'ulcer', 'tuberculosis', 'inflammation', 'granuloma', 'polyp',
        'thrombophlebitis', 'sebaceous', 'hypernephroma', 'phlebitis',
        'swelling', 'necrosis', 'osteomyelitis', 'ganglion', 'exostosis'
    ]

    abbreviation_map = {
        'ca.': 'carcinoma', 'ca': 'carcinoma',
        'tb': 'tuberculosis',
        'scc': 'squamous cell carcinoma'
    }

    institution_blacklist = [
        'the university of chicago', 'university clinics', 'laboratory of surgical pathology',
        'gross and microscopic description', 'operation', 'drawing', 'form no',
        'clinical diagnosis', 'pathological diagnosis', 'specimen', 'stored',
        'museum', 'destroyed'
    ]

    def expand_abbreviation(text):
        words = text.lower().split()
        return ' '.join([abbreviation_map.get(w.strip('.'), w) for w in words])

    def is_structural_label(line):
        return any(line.lower().strip().startswith(term) for term in structural_terms)

    def is_valid_diagnosis(text):
        if not text or len(text.strip()) < 3:
            return False

        text = expand_abbreviation(text)
        lowered = text.lower().strip()

        if any(header in lowered for header in institution_blacklist):
            return False

        if sum(1 for w in text.split() if w.isupper()) >= 3:
            return False

        invalid_starts = structural_terms + ['age of patient']
        if any(lowered.startswith(prefix) for prefix in invalid_starts):
            return False

        if any(key in lowered for key in ['slide #', 'does not match', 'removed in', 'transferred from', 'unit #']):
            return False

        if re.search(r'\b(dr\.?|surgeon|mr\.?|ms\.?|mrs\.?)\b', lowered):
            return False
        if re.search(r'\b[A-Z][a-z]+,?\s+[A-Z][a-z]+', text):
            return False
        if re.match(r'^[\d\s]+$', text):
            return False
        if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', text):
            return False
        if re.match(r'^[\.\s]*yrs[\.\s]*\.?$', lowered):
            return False
        if re.match(r'^(this is\s+)?no\.?\s*\d+', lowered):
            return False
        if lowered.startswith('this is') or lowered.endswith('no'):
            return False
        if re.match(r'^[\?\.\-:]', text.strip()):
            return False

        if len(text.split()) <= 2:
            if not any(word in lowered for word in medical_keywords):
                return False

        return True

    def extract_from_operation(lines):
        for line in lines:
            if 'operation' in line.lower() and ':' in line:
                value = line.split(':', 1)[-1].strip()
                if is_valid_diagnosis(value):
                    return value
        return ''

    for i, line in enumerate(lines):
        lower = line.lower().strip()

        if any(kw in lower for kw in field_keywords):

            parts = re.split(r'clinical(?: data and)? diagnosis[:.\-_\s]*', line, flags=re.IGNORECASE)
            if len(parts) > 1:
                value = parts[1].strip(" .,:;")
                if is_valid_diagnosis(value):
                    return value

            for j in range(1, 3):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    if not next_line:
                        continue
                    if is_structural_label(next_line):
                        break
                    if 'age of patient' in next_line.lower() or 'yrs' in next_line.lower():
                        continue
                    if is_valid_diagnosis(next_line):
                        return next_line

            candidate_lines = []
            for j in range(1, 6):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    if not next_line:
                        continue
                    if is_structural_label(next_line):
                        continue
                    if 'age of patient' in next_line.lower() or 'yrs' in next_line.lower():
                        continue
                    if is_valid_diagnosis(next_line):
                        candidate_lines.append(next_line)
            if candidate_lines:
                combined = ' '.join(candidate_lines)
                if is_valid_diagnosis(combined):
                    return combined.strip()

    fallback = extract_from_operation(lines)
    if fallback:
        return fallback

    return ''

def extract_path_diagnosis(text):

    DIAGNOSIS_KEYWORDS = [
        'cyst', 'fistula', 'carcinoma', 'tumor', 'abscess', 'ulcer',
        'polyp', 'osteomyelitis', 'tuberculosis', 'granuloma', 'sarcoma',
        'gangrene', 'hyperplasia', 'inflammation', 'lymphadenitis', 'fibroma',
        'pyogenic', 'neoplasm', 'nodes', 'necrosis', 'veins', 'empyema',
        'goitre', 'goiter', 'cholecystitis', 'mastitis', 'varicocele', 'varicocoele',
        'salpingitis', 'dacryocystitis', 'papilloma', 'epidermoid', 'hemangioma',
        'adenocarcinoma', 'papillary', 'scirrhus', 'epithelioma'
    ]
    
    ANATOMY_TERMS = [
        'connective tissue', 'epithelium', 'plasma cells', 'polymorphs',
        'fibroblasts', 'fatty tissue', 'capillary buds', 'lymphatics',
        'tubular glands', 'mucosa', 'glands', 'acini', 'hyperchromatism',
        'mitosis', 'stroma', 'blood vessels', 'cells are normal',
        'columnar cells', 'degeneration', 'spindle cells', 'fibrous tissue'
    ]
    
    ALLOWED_NORMAL_DIAGNOSES = [
        'normal appendix', 'normal tonsil', 'normal mucosa',
        'normal lymph node', 'normal tissue'
    ]
    
    ALLOWED_SINGLE_WORD = [
        'tuberculosis', 'empyema', 'carcinoma', 'gangrene', 'fibroma'
    ]
    
    EXCLUDE_PREFIXES = [
        'operation', 'stored', 'destroyed', 'museum', 'name', 'form', 'date',
        'gross', 'microscopic', 'section', 'interpretation', 'specimen',
        'clinical', 'cut surface', 'clinic', 'attending', 'removed from',
        'sent to', 'unit', 'blocks', 'index', 'only one slide', 'slide shows',
        'described above'
    ]
    
    NEGATIVE_PHRASES = [
        'not seen', 'no evidence', 'absent', 'negative for', 'does not show'
    ]
    
    MICROSCOPIC_HINTS = ['the section', 'the cells', 'there is', 'composed of']
    PATH_PREFIX = r'Path(?:\.|\w{0,20})?\s*Diagnosis'

    def clean_diagnosis(txt):
        txt = re.sub(r'\b(?:Path(?:\.|\w{0,20})?\s*)?Diagnosis\b[:.\-]?\s*', '', txt, flags=re.IGNORECASE)
        txt = txt.strip(" .:;\n")
        if ' with ' in txt and len(txt.split()) > 10:
            txt = txt.split(' with ')[0]
        return txt

    def is_valid(txt):
        if not txt or len(txt.strip()) < 3:
            return False
        lowered = txt.lower().strip()

        if any(neg in lowered for neg in NEGATIVE_PHRASES):
            return False
        if lowered in ALLOWED_NORMAL_DIAGNOSES or lowered in ALLOWED_SINGLE_WORD:
            return True
        if lowered.startswith('normal') and len(lowered.split()) <= 5:
            return True
        if any(lowered.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            return False
        if any(term in lowered for term in ANATOMY_TERMS):
            return False
        if len(lowered.split()) > 25:
            return False
        if any(term in lowered for term in DIAGNOSIS_KEYWORDS):
            return True
        return False

    lines = text.splitlines()
    diagnosis_lines = []

    for i, line in enumerate(lines):
        inline_match = re.search(fr'{PATH_PREFIX}[:.\-]?\s*(.+)', line, re.IGNORECASE)
        if inline_match:
            candidate = clean_diagnosis(inline_match.group(1).strip())
            if is_valid(candidate):
                return candidate
        elif re.search(PATH_PREFIX, line, re.IGNORECASE):
            for j in range(1, 5):
                if i + j < len(lines):
                    follow_line = lines[i + j].strip()
                    if follow_line:
                        diagnosis_lines.append(follow_line)
                    else:
                        break
            break

    if diagnosis_lines:
        merged = ' '.join(diagnosis_lines)
        merged = re.sub(r'[^A-Za-z0-9 ,.\-]', ' ', merged)
        merged = re.sub(r'\s+', ' ', merged).strip()
        merged = clean_diagnosis(merged)
        if is_valid(merged):
            return merged

    keyword_pattern = r'\b(?:' + '|'.join(DIAGNOSIS_KEYWORDS + ['normal']) + r')\b'
    fallback_candidates = []

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if lowered.startswith(('section', 'gross', 'microscopic')) or re.match(r'^\d+\.', stripped):
            continue
        if any(prefix in lowered for prefix in EXCLUDE_PREFIXES + MICROSCOPIC_HINTS):
            continue
        if any(neg in lowered for neg in NEGATIVE_PHRASES):
            continue

        words = lowered.split()
        if re.search(keyword_pattern, lowered):
            if len(words) == 1 and words[0] in DIAGNOSIS_KEYWORDS:
                fallback_candidates.append(words[0].capitalize())
                continue
            candidate = clean_diagnosis(stripped)
            if is_valid(candidate):
                fallback_candidates.append(candidate)

    if fallback_candidates:
        unique_parts = []
        for item in fallback_candidates:
            if all(item.lower() not in u.lower() for u in unique_parts):
                unique_parts.append(item)
        merged = '; '.join(unique_parts)
        if is_valid(merged):
            return merged

    return ''

def extract_operation(text):
    lines = text.splitlines()
    operation_lines = []
    collecting = False

    operation_start_keywords = ['operation', 'operation.']
    operation_content_starters = [
        'excision', 'removal', 'biopsy', 'appendectomy', 'cholecystectomy',
        'thyroidectomy', 'mastectomy', 'hysterectomy', 'laparotomy', 'resection',
        'amputation', 'ligation', 'curettage', 'radical', 'repair', 'polya',
        'exploratory', 'pyelotomy', 'dilation', 'incision', 'suspension',
        'nephrectomy', 'exenteration', 'tonsillectomy', 'cystotomy', 'enucleation',
        'hemiglossectomy', 'lobectomy', 'marsupialization', 'fusion', 'aspiration',
        'perineorrhaphy', 'hemorrhoidectomy', 'dacryocystectomy', 'oophorectomy'
    ]
    past_tense_patterns = [
        r'\b(removed|excised|amputated|resected|drained|implanted|extracted)\s+.*?(goiter|tumor|mass|cyst|appendix|fibroid|lymph|thyroid|gland|growth)',
    ]
    stop_keywords = [
        'gross', 'microscopic', 'clinical diagnosis', 'clinical data',
        'attending', 'clinic or floor', 'bacteriology', 'stored', 'museum',
        'destroyed', 'path. no', 'unit no', 'form no', 'no. of blocks', 'date'
    ]

    def clean_line(line):
        line = re.sub(r'(date of operation[:\-\s]*\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\b(date|unit no|form no|path\.? no|age)\b.*$', '', line, flags=re.IGNORECASE)
        line = re.sub(r'^(gross and microscopic description and drawing\.?\s*operation[:.\-\s]*)', '', line, flags=re.IGNORECASE)
        line = re.sub(r'^oberation[:.\-\s]*', '', line, flags=re.IGNORECASE)
        return line.strip(" .:-").capitalize()

    def is_truncated(line):
        return re.search(r'\b(and|or|with|about|of|from|on|in)$', line.strip().lower()) is not None

    def is_valid_operation_line(line):
        line_clean = line.strip().lower()
        if (
            len(line_clean) < 3 or
            re.fullmatch(r'[0-9\s:\.\-\/]+', line_clean) or
            not re.search(r'[a-zA-Z]', line_clean) or
            re.search(r'\b(dr\.?|mr\.?|ms\.?|mrs\.?|md)\b', line_clean) or
            re.match(r'^[A-Z][a-z]+,\s*[A-Z]', line.strip()) or
            re.match(r'^[A-Z][a-z]+[,]?$', line.strip())
        ):
            return False
        has_keyword = any(k in line_clean for k in operation_content_starters)
        is_standalone_keyword = line_clean in operation_content_starters
        return has_keyword or is_standalone_keyword

    i = 0
    while i < len(lines):
        line = lines[i]
        line_clean = line.strip().lower()

        if not collecting:
            if any(k == line_clean.strip() for k in operation_start_keywords):
                collecting = True

                for j in range(1, 5):
                    if i + j >= len(lines):
                        break
                    next_line = lines[i + j].strip()
                    if is_valid_operation_line(next_line) and not is_truncated(next_line):
                        operation_lines.append(clean_line(next_line))
                        break
                i += 1
                continue
            elif any(line_clean.startswith(k) for k in operation_content_starters):
                if is_valid_operation_line(line) and not is_truncated(line):
                    operation_lines.append(clean_line(line))
                    collecting = True
                i += 1
                continue
        elif any(k in line_clean for k in stop_keywords):
            break
        elif is_valid_operation_line(line) and not is_truncated(line):
            operation_lines.append(clean_line(line))
        i += 1

    if not operation_lines:
        for j in range(len(lines)):
            if any(k == lines[j].strip().lower() for k in operation_start_keywords):
                for k in range(j + 1, min(j + 4, len(lines))):
                    next_line = lines[k].strip()
                    if 'pathological diagnosis' in next_line.lower():
                        continue
                    if is_valid_operation_line(next_line) and not is_truncated(next_line):
                        operation_lines.append(clean_line(next_line))
                break

    operation_field_present = any('operation' in line.lower() for line in lines)
    if not operation_lines and not operation_field_present:
        for line in lines:
            if 'pathological diagnosis' in line.lower():
                for starter in operation_content_starters:
                    if starter in line.lower() and is_valid_operation_line(line):
                        content = line.split(":", 1)[-1]
                        if not is_truncated(content):
                            operation_lines.append(clean_line(content))
                        break
                break

    if not operation_lines:
        for line in lines:
            if is_valid_operation_line(line) and not is_truncated(line):
                operation_lines.append(clean_line(line))
                break

    if not operation_lines:
        for line in lines:
            for pattern in past_tense_patterns:
                match = re.search(pattern, line.lower())
                if match:
                    operation_lines.append(clean_line(match.group(0)))
                    break
            if operation_lines:
                break

    operation_lines = list(dict.fromkeys([line.capitalize() for line in operation_lines]))

    return ' '.join(operation_lines).strip() if operation_lines else ''

def extract_no_of_blocks(text):
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if re.search(r'\b(no\.?\s*of\s*blocks|blocks|blks)\b[:.\-]?', line, flags=re.IGNORECASE):
            match = re.search(r'\b(?:blocks|blks|no\.?\s*of\s*blocks)[:.\-]?\s*(\d+)\b', line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.search(r'\b(unit|path|form|no\.?)\b', next_line, re.IGNORECASE):
                    continue

                if re.fullmatch(r'\d+', next_line):
                    return next_line.strip()

    return ''

def extract_embedding_medium(text, keywords=None):
    if keywords is None:
        keywords = [
            'paraffin', 'parafin', 'paraphin', 'paraffne', 'parafine',
            'celloidin', 'celliodin', 'celloidine', 'cellodin',
            'wax', 'waxed', 'resin', 'gelatin'
        ]

    def clean_line(line):
        return re.sub(r'[^a-z0-9\s\-:.]', '', line.lower().strip())

    lines = text.splitlines()
    candidates = []
    cleaned_lines = [clean_line(line) for line in lines]

    for i, line in enumerate(cleaned_lines):
        if re.search(r'(embedding\s*medium|embedding|medium|bacteriology)[\s\-:.\u2022]*$', line):
            for j in range(1, 4):
                if i + j < len(cleaned_lines):
                    follow = cleaned_lines[i + j]
                    for kw in keywords:
                        if kw in follow:
                            candidates.append(kw.capitalize())

        for kw in keywords:
            if re.search(rf'(embedding|medium|bacteriology)[\s\-:.\u2022]*{kw}', line):
                candidates.append(kw.capitalize())

        if line.strip() in keywords:
            candidates.append(line.strip().capitalize())

    return '; '.join(dict.fromkeys(candidates)) if candidates else ''

def extract_disposal_of_tissue(text):
    lines = text.splitlines()
    results = []

    KEYWORDS = {'stored', 'museum', 'destroyed'}
    FUZZY_MAP = {
        'stared': 'Stored',
        'storod': 'Stored',
        'destroyd': 'Destroyed',
        'destoryed': 'Destroyed',
        'musuem': 'Museum',
        'museun': 'Museum'
    }

    for i, line in enumerate(lines):
        if re.search(r'disposal\s*of\s*tissue\s*[:\-]?', line, flags=re.IGNORECASE):
            for j in range(1, 6):
                if i + j >= len(lines):
                    break
                candidate = lines[i + j].strip().lower()
                if not candidate:
                    continue

                if candidate in KEYWORDS:
                    results.append(candidate.capitalize())
                    continue
                elif candidate in FUZZY_MAP:
                    results.append(FUZZY_MAP[candidate])
                    continue

                for word in re.split(r'[\s,;/]+', candidate):
                    w = word.strip(string.punctuation).lower()
                    if w in KEYWORDS:
                        results.append(w.capitalize())
                    elif w in FUZZY_MAP:
                        results.append(FUZZY_MAP[w])
            break

    return '; '.join(dict.fromkeys(results)) if results else ''

def extract_gross_description(text, debug=False):
    gross_start_patterns = [
        r'^\s*(g|r)?ross\s+descrip(?:tion|toin|ti[ao]n|tio[nm])\s*[:\-]?\s*',
        r'^\s*(g|r)?ross\s+and\s+microscopic\s+description\s*(and\s+drawing)?\s*[:\-]?\s*',
        r'^\s*:?\s*gross\s*[:\-]?\s*$',
        r'^\s*gross[\s\-:]',
        r'^\s*GROSS[\s\-:]',
    ]

    end_keywords = [
        r'^microscopic\b', r'^microscopy\b', r'^clinical diagnosis', r'^final diagnosis',
        r'^pathological diagnosis', r'^disposal of tissue', r'^bacteriology',
        r'^stored', r'^destroyed', r'^blocks\b', r'^celloidin', r'^notes\b',
        r'^operation\b', r'^interpretation', r'^index\b', r'^summary of case',
        r'^\s*section\b', r'^specimen\b', r'^report\b'
    ]

    lines = text.splitlines()
    paragraphs = []
    current_para = []
    in_gross = False
    start_line_no = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_gross:
            for pattern in gross_start_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    in_gross = True
                    start_line_no = i
                    cleaned_line = re.sub(pattern, '', stripped, flags=re.IGNORECASE).strip()
                    if cleaned_line:
                        current_para.append(cleaned_line)
                    break
            continue

        if in_gross:
            if any(re.match(k, stripped, re.IGNORECASE) for k in end_keywords):
                if current_para:
                    if debug:
                        print(f"[Gross paragraph from line {start_line_no + 1} to {i}]")
                        print("→", ' '.join(current_para)[:120], "...\n")
                    paragraphs.append(' '.join(current_para).strip())
                    current_para = []
                in_gross = False
                continue

            if len(current_para) > 100:
                if debug:
                    print(f"[Gross paragraph auto-truncated at line {i}]")
                paragraphs.append(' '.join(current_para).strip())
                current_para = []
                in_gross = False
                continue

            current_para.append(stripped)

    if in_gross and current_para:
        if debug:
            print(f"[Gross paragraph ends at EOF from line {start_line_no + 1}]")
            print("→", ' '.join(current_para)[:120], "...\n")
        paragraphs.append(' '.join(current_para).strip())

    return '\n\n'.join(paragraphs)

def extract_microscopic_description(text):
    lines = text.splitlines()
    desc_lines = []
    capture = False

    start_patterns = [
        r"^\s*Microscopic[:\.]?",
        r"^\s*Microscopic sections",
        r"^\s*Microscopically",
        r"^\s*Histologic (sections|findings)",
        r"^\s*Microscopic examination",
        r"^\s*MICROSCOPIC DESCRIPTION[:\.]?"
    ]

    stop_patterns = [
        r"^\s*[A-Z\s]{5,}$",
        r"^\s*Gross\s*[:\.]?",           
        r"^\s*Diagnosis\s*[:\.]?",       
        r"^\s*Final Diagnosis",
        r"^\s*Index[:\.]?",              
        r"^\s*Specimen[:\.]?",           
        r"^\s*Clinical[:\.]?",           
        r"^\s*Operation[:\.]?",          
        r"^\s*Pathologic[:\.]?",         
        r"^\s*Name[-:]?",                
        r"^\s*Drawing[:\.]?",            
        r"^\s*[A-Z]\.\s*[A-Z]\."
    ]

    for line in lines:
        if not capture:
            for pat in start_patterns:
                if re.match(pat, line, re.IGNORECASE):
                    capture = True
                    desc_lines.append(line.strip())
                    break
        elif capture:
            if any(re.match(pat, line, re.IGNORECASE) for pat in stop_patterns):
                break
            desc_lines.append(line.strip())

    raw_text = ' '.join(desc_lines).strip()
    fixed_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', raw_text)

    if len(fixed_text.split()) < 5 or fixed_text.lower().startswith("microscopic") and len(desc_lines) < 2:
        return ""

    return fixed_text

def extract_index_line(text):
    lines = text.splitlines()
    index_entries = []

    for line in lines:
        matches = re.findall(r"\bIndex\s*[:\-–—]\s*.*?(?=(?:\bIndex\s*[:\-–—]|$))", line, re.IGNORECASE)
        for match in matches:
            entry = match.strip().rstrip(":.-–— ").replace("–", "-").replace("—", "-")
            if entry and entry.lower() not in {"index", "index:"} and len(entry.split()) >= 2:
                index_entries.append(entry)

    unique_entries = list(dict.fromkeys(index_entries))

    return ' | '.join(unique_entries).strip() if unique_entries else ""

def extract_full_description(text):
    lines = text.splitlines()
    description_lines = []
    capture = False

    start_patterns = [
        r"^\s*Microscopic\s*[:\.]?",
        r"^\s*Microscopic Description\s*[:\.]?",
        r"^\s*Microscopic sections",
        r"^\s*Gross\s*[:\.]?",
        r"^\s*Clinical\s*Data\s*[:\.]?",
        r"^\s*Clinical\s*[:\.]?",
        r"^\s*Final\s*Diagnosis\s*[:\.]?",
        r"^\s*Diagnosis\s*[:\.]?",
        r"^\s*Histologic (findings|sections)",
        r"^\s*Pathologic (description|report|examination)",
    ]

    stop_patterns = [
        r"^\s*Index\s*[:\-–—]?",
        r"^\s*Name\s*[-:]?",
        r"^\s*Date\s*[:\-]?",
        r"^\s*Unit No\.?",
        r"^\s*Drawing\s*[:\.]?",
        r"^\s*OPERATION",
        r"^\s*[A-Z\s]{5,}$",
    ]

    for line in lines:
        line_clean = line.strip()

        if not capture:
            if any(re.match(pat, line_clean, re.IGNORECASE) for pat in start_patterns):
                capture = True
                description_lines.append(line_clean)
        else:
            if any(re.match(pat, line_clean, re.IGNORECASE) for pat in stop_patterns):
                break
            description_lines.append(line_clean)

    result = ' '.join(description_lines).strip()
    return result if len(result.split()) > 5 else ""

def extract_raw_text(text):
    return text.strip()

def extract_path_no(text):
    lines = text.splitlines()
    candidates = []

    for i, line in enumerate(lines):
        line_clean = line.strip()
        line_lower = line_clean.lower()

        if 'path' in line_lower and 'no' in line_lower:
            for j in range(3):
                if i + j < len(lines):
                    target_line = lines[i + j].strip()
                    if 'form no' in target_line.lower():
                        continue
                    nums = re.findall(r'\d{3,6}', target_line)
                    candidates.extend(nums)

        elif i + 1 < len(lines):
            next_line = lines[i + 1].lower()
            if 'path' in next_line and 'no' in next_line:
                nums = re.findall(r'\d{3,6}', line_clean)
                candidates.extend(nums)

        nums = re.findall(r'\d{3,6}', line_clean)
        if nums and 'form no' not in line_lower:
            candidates.extend(nums)

    filtered = [n for n in candidates if is_valid_path_no(n)]
    if not filtered:
        return ''

    for n in reversed(filtered):
        if len(n) == 4 and n.startswith('1'):
            return n
    for n in reversed(filtered):
        if len(n) == 4:
            return n
    return filtered[-1]

def is_valid_path_no(n):
    if not n.isdigit():
        return False
    if n == '812':
        return False
    if len(n) < 3 or len(n) > 6:
        return False
    if len(n) == 5 and int(n[:2]) > 20:
        return False
    if len(n) == 4 and int(n[:2]) in {18, 19, 20, 21}:
        return False
    return True

def extract_paragraph(text, start_kw, end_kw=None):
    if end_kw:
        pattern = rf"{start_kw}[.:]*\s*(.+?)\s*{end_kw}"
    else:
        pattern = rf"{start_kw}[.:]*\s*(.+)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ''

def is_meaningful(value):
    if not value or not isinstance(value, str):
        return False
    value = value.strip()
    return not re.fullmatch(r'[-. ]*', value) and len(value) > 2

def extract_year_from_filename(fname):
    match = re.search(r'(\d{4})', fname)
    return int(match.group(1)) if match else None

def parse_ocr_text(text, year=None):
    record = {}
    record['form_no'] = extract_form_no(text)
    institution, department = extract_institution_and_department(text)
    record['institution'] = institution
    record['department'] = department

    record['name'] = extract_patient_name(text)
    record['gender'] = guess_gender(record['name']) if record['name'] else ''
    record['age'] = extract_age(text)

    record['unit_no'] = extract_unit_no(text)
    record['path_no'] = extract_path_no(text)
    record['date'] = extract_date(text)
    record['clinic_or_floor'] = extract_clinic_or_floor(text)

    record['attending_physician'] = extract_attending_physician(text)
    record['surgeon'] = extract_surgeon(text)

    record['clinical_diagnosis'] = extract_clinical_diagnosis(text)
    record['path_diagnosis'] = extract_path_diagnosis(text)
    record['operation'] = extract_operation(text)

    record['no_of_blocks'] = extract_no_of_blocks(text)

    record['embedding_medium'] = extract_embedding_medium(text)

    record['disposal_of_tissue'] = extract_disposal_of_tissue(text)

    record['gross_description'] = extract_gross_description(text)
    record['microscopic_description'] = extract_microscopic_description(text)
    record['index_line'] = extract_index_line(text)
    record['full_description'] = extract_full_description(text)
    record['raw_text'] = extract_raw_text(text)

    for k in record:
        if isinstance(record[k], str):
            record[k] = record[k].replace('\n', ' ').replace('\r', '').strip()

    return record

def run_pipeline(input_dir, output_excel):
    records = []
    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])
    for idx, fname in enumerate(txt_files):
        with open(os.path.join(input_dir, fname), 'r', encoding='utf-8') as f:
            text = f.read().strip()
            year = extract_year_from_filename(fname)
            record = parse_ocr_text(text, year=year)
            record['index'] = idx + 1
            record['filename'] = fname

            exclude_keys = ['index', 'filename', 'status']
            has_content = any(is_meaningful(record.get(k, '')) for k in record if k not in exclude_keys)
            record['status'] = 'parsed' if has_content else 'empty'

            records.append(record)

    df = pd.DataFrame(records)

    final_cols = [
        'index', 'form_no', 'institution', 'department',
        'name', 'gender', 'age', 'unit_no', 'path_no', 
        'date', 'clinic_or_floor', 'attending_physician', 'surgeon',
        'clinical_diagnosis', 'path_diagnosis', 'operation',
        'no_of_blocks', 'embedding_medium', 'disposal_of_tissue',
        'gross_description', 'microscopic_description', 'index_line', 
        'full_description' , 'raw_text' ,'filename', 'status'
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = ''
    df = df[final_cols]
    csv_path = output_excel.replace('.xlsx', '.csv')
    df = fill_missing_unit_no(df, name_col='name', unit_col='unit_no')
    df.to_csv(csv_path, index=False)
    df.to_excel(output_excel, index=False)

if __name__ == "__main__":
    input_dir = "1928 txts"
    output_excel = "1928 Excel/ocr_1928_output.xlsx"
    run_pipeline(input_dir, output_excel)
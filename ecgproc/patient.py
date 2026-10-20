"""
Age and sex for the model.

Every record gets its age and sex from the first of these that has them,
field by field:

    1. the file itself      DICOM PatientAge or birth date + exam date, MAT
                            age/sex, WFDB header comments ("Age: 74",
                            "Sex: Male"), HL7 aECG / vendor XML demographics
    2. the metadata CSV     an ID column plus age and/or sex columns; a row
                            matches a record by the file name (with or without
                            extension), the patient ID stored in the file, or
                            the leading number of the file name. Record-level
                            ID columns (record_id, ecg_id, filename) match file
                            names only, patient-level ones (patient_id,
                            subject_id) the stored patient ID only.
    3. the fallback values  set for the run (GUI fields, --age / --sex,
                            config.ini [patient])

Sex is accepted as text only (M/F, male/female, man/woman, German
männlich/weiblich/w).
"""
import re
from datetime import date
from pathlib import Path

MALE = "Male"
FEMALE = "Female"

_SEX_WORDS = {
    "m": MALE, "male": MALE, "man": MALE, "mann": MALE, "männlich": MALE,
    "maennlich": MALE, "masculine": MALE,
    "f": FEMALE, "female": FEMALE, "woman": FEMALE, "w": FEMALE, "frau": FEMALE,
    "weiblich": FEMALE, "feminine": FEMALE,
}

# Unit suffix
_AGE_UNITS = {
    "": 1.0, "y": 1.0, "yr": 1.0, "yrs": 1.0, "year": 1.0, "years": 1.0, "a": 1.0,
    "j": 1.0, "jahre": 1.0,
    "m": 1 / 12, "mo": 1 / 12, "mos": 1 / 12, "mon": 1 / 12, "mth": 1 / 12, "mths": 1 / 12,
    "month": 1 / 12, "months": 1 / 12, "monat": 1 / 12, "monate": 1 / 12,
    "w": 7 / 365.25, "wk": 7 / 365.25, "wks": 7 / 365.25, "week": 7 / 365.25, "weeks": 7 / 365.25,
    "woche": 7 / 365.25, "wochen": 7 / 365.25,
    "d": 1 / 365.25, "day": 1 / 365.25, "days": 1 / 365.25, "tag": 1 / 365.25, "tage": 1 / 365.25,
}

MIN_AGE, MAX_AGE = 0.0, 125.0


def parse_sex(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return _SEX_WORDS.get(text)


def parse_age(value):
    if value is None:
        return None
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.item()
    except ImportError:
        pass
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        years = float(value)
    else:
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*([a-zA-Z]*)\s*$", str(value))
        if not m:
            return None
        unit = m.group(2).lower()
        if unit not in _AGE_UNITS:
            return None
        years = float(m.group(1).replace(",", ".")) * _AGE_UNITS[unit]
    if years != years or not (MIN_AGE <= years <= MAX_AGE):   # NaN or out of range
        return None
    return round(years, 2)


def parse_date(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.fullmatch(r"(\d{8})(?:\d{2,6})?(?:\.\d+)?(?:[+-]\d{4})?", text)
    if m:
        text = m.group(1)
    digits = re.sub(r"\D", "", text)
    parts = [p for p in re.split(r"[-./\s]", text.split("T")[0]) if p]
    try:
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            a, b, c = parts
            if len(a) == 4:                                  # Y-M-D
                return date(int(a), int(b), int(c))
            if len(c) == 4:
                sep = re.search(r"[-./]", text)
                if sep and sep.group(0) == ".":              # D.M.Y (German)
                    return date(int(c), int(b), int(a))
                x, y = int(a), int(b)
                if x > 12 >= y:                              # D/M/Y
                    return date(int(c), y, x)
                if y > 12 >= x:                              # M/D/Y
                    return date(int(c), x, y)
                return None                                  # ambiguous
            return None
        if len(digits) >= 8 and len(parts) <= 1:
            yyyy = int(digits[:4])
            if 1880 <= yyyy <= 2100:                         # YYYYMMDD[HHMMSS]
                return date(yyyy, int(digits[4:6]), int(digits[6:8]))
            day, month = int(digits[0:2]), int(digits[2:4])
            if day > 12 or day == month:                     # DDMMYYYY, not MMDDYYYY
                return date(int(digits[4:8]), month, day)
            return None                                      # ambiguous
    except (ValueError, OverflowError):
        return None
    return None


def age_from_dates(birth, exam):
    b, e = parse_date(birth), parse_date(exam)
    if b is None or e is None or e < b:
        return None
    years = e.year - b.year - ((e.month, e.day) < (b.month, b.day))
    return parse_age(years)

_COMMENT_AGE = re.compile(
    r"\bage>?\s*[:=]\s*([0-9]+(?:[.,][0-9]+)?)([A-Za-z]*)(?:\s+([A-Za-zäöüÄÖÜ]+)(\s*[:=])?)?",
    re.IGNORECASE)
_COMMENT_SEX = re.compile(r"\b(?:sex|gender)>?\s*[:=]\s*([A-Za-zäÄ]+)", re.IGNORECASE)


def from_file(meta):
    meta = meta or {}
    age, age_how = parse_age(meta.get("patient_age")), "age field"
    if age is None:
        age, age_how = age_from_dates(meta.get("patient_birthdate"), meta.get("exam_date")), "birth date"
    sex, sex_how = parse_sex(meta.get("patient_sex")), "sex field"

    comments = " ; ".join(str(c) for c in (meta.get("comments") or []))
    if comments:
        if age is None:
            m = _COMMENT_AGE.search(comments)
            age, age_how = (_comment_age(*m.groups()) if m else None), "header comment"
        if sex is None:
            m = _COMMENT_SEX.search(comments)
            sex, sex_how = (parse_sex(m.group(1)) if m else None), "header comment"

    return (age, age_how if age is not None else "",
            sex, sex_how if sex is not None else "")


def _comment_age(number, attached, word, label):
    """Age from a header comment's parts (see _COMMENT_AGE); None when the unit is unknown."""
    if attached:                                 
        return parse_age(number + attached)
    if word and not label and len(word) > 1 and word.lower() not in ("sex", "gender"):
        return parse_age(f"{number} {word}") if word.lower() in _AGE_UNITS else None
    return parse_age(number)                     


# Metadata CSV


def _norm_header(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def normalize_id(value):
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        return None
    m = re.fullmatch(r"([+-]?)(\d+)(?:\.0*)?", text)
    if m:                       
        digits = m.group(2).lstrip("0") or "0"
        return ("-" if m.group(1) == "-" and digits != "0" else "") + digits
    try:
        num = float(text)
        if num == int(num):
            return str(int(num))
    except (ValueError, OverflowError):
        pass
    return text.lower()


# Header spellings
_GENERIC_ID = ("id",)
_RECORD_ID = ("recordid", "record", "recordname", "ecgid", "ecg", "filename", "file", "studyid")
_PATIENT_ID = ("patientid", "patient", "subjectid", "subject")
_AGE = ("age", "basisage", "patientage")
_SEX = ("sex", "gender", "basissex", "patientsex")


class MetadataTable:
    """
    Age and sex per ID from the metadata CSV, kept apart by what the ID names:

        record   a record-level column (record_id, ecg_id, filename, ...): matched
                 against the file name, with or without extension, and the
                 leading number of the file name
        generic  a plain ID column: matched against the file name, the patient ID
                 stored in the file, and the leading number of the file name
        patient  a patient-level column (patient_id, subject_id): matched only
                 against the patient ID stored in the file

    so a record number can never pick up the row of a patient that happens to
    carry the same number.
    """

    def __init__(self, name, columns):
        self.name = name
        self.columns = columns
        self.record, self.generic, self.patient = {}, {}, {}

    def __len__(self):
        return max(len(self.record), len(self.generic), len(self.patient))

    def lookup(self, file_path, patient_id="", record_id=None):
        p = Path(file_path)
        m = re.match(r"(\d+)", p.stem)
        number = m.group(1) if m else None
        names = []
        if record_id:
            rid = str(record_id).replace("\\", "/")
            names += [rid, rid[: -len(Path(rid).suffix)] if Path(rid).suffix else rid]
        if p.suffix.lower() in (".dat", ".hea"):
            names += [p.with_suffix(".hea").name, p.with_suffix(".dat").name]
        names += [p.stem, p.name]
        names = list(dict.fromkeys(names))
        attempts = ([(self.record, "file name", n) for n in names]
                    + [(self.record, "file number", number)]
                    + [(self.generic, "file name", n) for n in names]
                    + [(self.generic, "patient ID", patient_id), (self.generic, "file number", number),
                       (self.patient, "patient ID", patient_id)])
        for table, kind, raw in attempts:
            key = normalize_id(raw)
            if key is not None and key in table:
                age, sex = table[key]
                return age, sex, kind, key
        return None


def load_metadata_csv(path, log=print):
    import pandas as pd

    try:
        with open(path, "rb") as fh:
            bom = fh.read(2)
    except OSError as e:
        log(f"Failed to read the metadata CSV {path}: {e}")
        return None
    encodings = ("utf-16",) if bom in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=encoding)
            break
        except UnicodeError:
            continue
        except Exception as e:
            log(f"Failed to read the metadata CSV {path}: {e}")
            return None
    else:
        log(f"Failed to read the metadata CSV {path}: unknown text encoding (save it as UTF-8).")
        return None

    by_norm = {}
    for col in df.columns:
        by_norm.setdefault(_norm_header(col), col)
    for col in df.columns:
        m = re.search(r"[(\[{]([^)\]}]*)[)\]}]\s*$", str(col))
        if not m:
            continue
        bare, unit = _norm_header(str(col)[:m.start()]), _norm_header(m.group(1))
        if not bare:
            bare, unit = unit, ""
        if not bare or (bare in _AGE and unit not in ("", "y", "yr", "yrs", "year", "years", "a",
                                                       "j", "jahre")):
            continue
        by_norm.setdefault(bare, col)

    def first(names):
        return next((by_norm[n] for n in names if n in by_norm), None)

    columns = {kind: first(names) for kind, names in
               (("record", _RECORD_ID), ("generic", _GENERIC_ID), ("patient", _PATIENT_ID))}
    columns = {k: v for k, v in columns.items() if v is not None}
    age_col, sex_col = first(_AGE), first(_SEX)
    if not columns or not (age_col or sex_col):
        log("Metadata CSV needs an ID column (ID, record_id, ecg_id, filename, patient_id, ...) "
            f"plus an age and/or sex column (found: {', '.join(map(str, df.columns))}). Not used.")
        return None

    table = MetadataTable(Path(path).name, columns)
    bad_age = bad_sex = 0
    conflicts = set()
    for _, row in df.iterrows():
        age = parse_age(row.get(age_col)) if age_col else None
        sex = parse_sex(row.get(sex_col)) if sex_col else None
        if age_col and age is None and str(row.get(age_col) or "").strip() not in ("", "nan"):
            bad_age += 1
        if sex_col and sex is None and str(row.get(sex_col) or "").strip() not in ("", "nan"):
            bad_sex += 1
        for kind, col in columns.items():
            key = normalize_id(row.get(col))
            if key is None:
                continue
            ids = getattr(table, kind)
            if key in ids:
                old_age, old_sex = ids[key]
                if (age is not None and old_age is not None and age != old_age) or \
                        (sex is not None and old_sex is not None and sex != old_sex):
                    conflicts.add((kind, key))
                age_kept = age if age is not None else old_age
                sex_kept = sex if sex is not None else old_sex
                ids[key] = (age_kept, sex_kept)
            else:
                ids[key] = (age, sex)
    how = {"record": "by file name", "generic": "by file name or stored patient ID",
           "patient": "by stored patient ID"}
    used = "; ".join(f"'{col}' {how[kind]}" for kind, col in columns.items())
    values = " and ".join(f"{what} from '{col}'" for what, col in (("age", age_col), ("sex", sex_col)) if col)
    log(f"Metadata CSV: {len(table)} rows from {table.name}, matched {used}; {values}.")
    for what, col, names in (("age", age_col, "age, basis_age, patient_age"),
                             ("sex", sex_col, "sex, gender, basis_sex, patient_sex")):
        if not col:
            log(f"Metadata CSV: no {what} column (looked for {names}; found: "
                f"{', '.join(map(str, df.columns))}).")
    if conflicts:
        log(f"Metadata CSV: {len({key for _, key in conflicts})} ID(s) appear in several rows with "
            "different age or sex; later rows override earlier ones, blank cells keep earlier values.")
    if bad_age:
        log(f"Metadata CSV: {bad_age} age value(s) not understood (need years, e.g. 63 or 063Y).")
    if bad_sex:
        log(f"Metadata CSV: {bad_sex} sex value(s) not understood (need M/F, male/female; "
            "numeric codes are not accepted).")
    return table


def resolve(meta, file_path, table=None, fallback_age=None, fallback_sex=None, reveal_ids=True,
            record_id=None):
    age, age_how, sex, sex_how = from_file(meta)
    out = {"age": age, "age_source": "file" if age is not None else "", "age_detail": age_how,
           "sex": sex, "sex_source": "file" if sex is not None else "", "sex_detail": sex_how}

    if table is not None and (age is None or sex is None):
        hit = table.lookup(file_path, (meta or {}).get("patient_id", ""), record_id=record_id)
        if hit is not None:
            c_age, c_sex, kind, key = hit
            detail = f"{kind} {key}" if reveal_ids else kind
            if out["age"] is None and c_age is not None:
                out.update(age=c_age, age_source="csv", age_detail=detail)
            if out["sex"] is None and c_sex is not None:
                out.update(sex=c_sex, sex_source="csv", sex_detail=detail)

    if out["age"] is None and fallback_age is not None:
        out.update(age=parse_age(fallback_age), age_source="fallback", age_detail="run setting")
    if out["sex"] is None and fallback_sex is not None:
        out.update(sex=parse_sex(fallback_sex), sex_source="fallback", sex_detail="run setting")
    for field in ("age", "sex"):
        if out[field] is None:
            out[f"{field}_source"] = ""
            out[f"{field}_detail"] = ""
    return out
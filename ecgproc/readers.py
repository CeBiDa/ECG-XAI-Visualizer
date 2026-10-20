import re
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import wfdb
import pydicom
import scipy.io as sio


from ecgproc.xml_utils import parse_xml, find_ecg_rhythms, extract_rhythms_snippet, parse_samples, find_first_tag_text, detect_sampling_frequency, lxml_etree
from ecgproc.constants import FINAL_LEADS
from ecgproc.files import zero_pad

# MAT Reader
def read_mat(file_path):
    try:
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError as e:        # scipy's 'use HDF reader for v7.3' advice
        raise ValueError(f"{Path(file_path).name} is a MATLAB v7.3 (HDF5) file, which cannot be "
                         "read; save it in MATLAB with save(..., '-v7').") from e
    signals = None
    if "template" in mat:
        signals = mat["template"]
    elif "val" in mat:
        signals = mat["val"]
    elif "ecg" in mat:
        signals = mat["ecg"]

    if signals is None:
        raise ValueError(f"No ECG signals found in {file_path} (checked for 'template', 'val', 'ecg')")

    def _get_scalar(data, key):
        if key in data:
            item = data[key]
            if hasattr(item, 'value'):
                return item.value
            return item
        return None

    if signals.ndim != 2:
        raise ValueError(f"MAT signals must be 2D, got shape {signals.shape}")
    if signals.shape[0] > 16 and signals.shape[1] <= 16:
        signals = signals.T

    n_leads = signals.shape[0]

    # Create signal names
    if n_leads == 12:
        signal_names = FINAL_LEADS.copy()
    else:
        signal_names = [f"lead{i+1}" for i in range(n_leads)]

    # Extract metadata fields
    fs = _get_scalar(mat, 'fs')
    patient_id = _get_scalar(mat, 'study_id')
    age = _get_scalar(mat, 'age')
    sex = _get_scalar(mat, 'sex')
    pon = _get_scalar(mat, 'pon')
    poff = _get_scalar(mat, 'poff')
    pdur = _get_scalar(mat, 'pdur')

    meta = {
        "source": "mat",
        "signal_names": signal_names,
        "n_samples": signals.shape[1],
        "fs": fs,
        "patient_id": str(patient_id) if patient_id is not None else "",
        "patient_firstname": "",
        "patient_lastname": "",
        "patient_birthdate": "",
        "patient_age": age,
        "patient_sex": str(sex) if sex is not None else "",
        "exam_date": "",
        "exam_time": "",
        "pon": pon,
        "poff": poff,
        "pdur": pdur
    }
    return signals, meta

# WFDB Reader
def read_dat_hea(file_path):
    file_path = Path(file_path)
    record_name = str(file_path.with_suffix("")) if file_path.suffix.lower() in (".dat", ".hea") \
        else str(file_path)
    header = Path(record_name + ".hea")
    if not header.exists() and header.parent.is_dir():
        variant = next((p for p in header.parent.iterdir()
                        if p.name.lower() == header.name.lower()), None)
        if variant is not None:
            return _read_wfdb_case_variant(variant)
    record = wfdb.rdrecord(record_name)
    if record.p_signal is None:
        raise ValueError(f"WFDB record {record_name} carries no sample data.")
    signals = record.p_signal.T

    def _stamp(value, fmt):
        if value in (None, ""):
            return ""
        return value.strftime(fmt) if hasattr(value, "strftime") else str(value)

    meta = {
        "source": "wfdb",
        "fs": record.fs,
        "signal_names": list(record.sig_name or []),
        "n_samples": signals.shape[1],
        "patient_id": str(getattr(record, "patient_id", "") or ""),
        "patient_firstname": "",
        "patient_lastname": "",
        "patient_birthdate": "",
        "exam_date": _stamp(getattr(record, "base_date", ""), "%Y%m%d"),
        "exam_time": _stamp(getattr(record, "base_time", ""), "%H%M%S"),
        "comments": [str(c) for c in (getattr(record, "comments", None) or [])],
    }
    return signals, meta

def _read_table(file_path):
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
        rows = [line for line in (fh.readline() for _ in range(5)) if line.strip()]
    kw = dict(encoding="utf-8-sig", encoding_errors="replace")
    decimal = "," if any("," in row for row in rows) else "."
    bare = re.sub(r'"[^"]*"', "", header)
    counts = {d: bare.count(d) for d in (",", ";", "\t")}
    sep = max(counts, key=counts.get)
    if counts[sep] and sep != ",":
        return pd.read_csv(file_path, sep=sep, decimal=decimal, **kw)
    names = header.split()
    if not counts[","] and len(names) > 1 and rows and len(rows[0].split()) == len(names):
        return pd.read_csv(file_path, sep=r"\s+", engine="python", decimal=decimal, **kw)
    return pd.read_csv(file_path, **kw)

def _read_wfdb_case_variant(header):
    stem = header.name[: -len(header.suffix)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / f"{stem}.hea").symlink_to(header)
        lines = [ln.split()[0] for ln in header.read_text(errors="replace").splitlines()[1:]
                 if ln.strip() and not ln.startswith("#")]
        for name in dict.fromkeys(lines):
            target = header.parent / name
            if not target.exists():
                target = next((p for p in header.parent.iterdir() if p.name.lower() == name.lower()),
                              target)
            if not (tmp / name).exists():
                (tmp / name).symlink_to(target)
        return read_dat_hea(tmp / f"{stem}.hea")

# CSV Reader
def read_csv(file_path):
    df = _read_table(file_path)
    try:
        [float(c) for c in df.columns]
        is_numeric_header = True
    except (ValueError, TypeError):
        is_numeric_header = False
    if is_numeric_header:
        raise ValueError(f"CSV {file_path} looks header-less (columns parse as numbers); "
                         "add a lead-name header row (e.g. I,II,III,aVR,aVL,aVF,V1..V6).")
    meta = {
        "source": "csv",
        "signal_names": list(df.columns),
        "n_samples": df.shape[0],
        "fs": None,
        "patient_id": "",
        "patient_firstname": "",
        "patient_lastname": "",
        "patient_birthdate": "",
        "exam_date": "",
        "exam_time": ""
    }
    return df.values.T, meta

# ASC Reader
def read_asc(file_path):
    df = _read_table(file_path)
    try:
        [float(c) for c in df.columns]
        is_numeric_header = True
    except (ValueError, TypeError):
        is_numeric_header = False
    if is_numeric_header:
        raise ValueError(f"ASC {file_path} looks header-less (columns parse as numbers); "
                         "add a lead-name header row.")
    signals = df.values.T # shape (n_leads, n_samples)
    meta = {
        "source": "asc",
        "signal_names": list(df.columns),
        "n_samples": signals.shape[1],
        "fs": None,
        "patient_id": "",
        "patient_firstname": "",
        "patient_lastname": "",
        "patient_birthdate": "",
        "exam_date": "",
        "exam_time": ""
    }
    return signals, meta

# DICOM Reader
def read_dcm(file_path):
    ds = pydicom.dcmread(file_path)
    
    # Parse Name
    patient_name = getattr(ds, "PatientName", "")
    firstname = ""
    lastname = ""
    if patient_name:
        if hasattr(patient_name, "given_name"):
            firstname = str(patient_name.given_name)
        if hasattr(patient_name, "family_name"):
            lastname = str(patient_name.family_name)
        else:
            parts = str(patient_name).split('^')
            if len(parts) > 0: lastname = parts[0]
            if len(parts) > 1: firstname = parts[1]

    # Extract Date
    def _get_exam_date(dataset):
        if "AcquisitionDateTime" in dataset:
            dt = str(dataset.AcquisitionDateTime)
            if len(dt) >= 8: return dt[:8]
            
        for tag in ["AcquisitionDate", "ContentDate", "SeriesDate", "StudyDate"]:
            if tag in dataset:
                val = str(getattr(dataset, tag))
                if val: return val[:8]
        return ""

    # Extract Time
    def _get_exam_time(dataset):
        if "AcquisitionDateTime" in dataset:
            dt = str(dataset.AcquisitionDateTime)
            if len(dt) >= 14: return dt[8:14]
            
        for tag in ["AcquisitionTime", "ContentTime", "SeriesTime", "StudyTime"]:
            if tag in dataset:
                val = str(getattr(dataset, tag))
                if val: return val.split('.')[0][:6]
        return ""

    meta = {
        "source": "dcm",
        "patient_id": str(getattr(ds, "PatientID", "")),
        "patient_firstname": firstname,
        "patient_lastname": lastname,
        "patient_birthdate": str(getattr(ds, "PatientBirthDate", "")),
        "patient_sex": str(getattr(ds, "PatientSex", "")),
        "patient_age": str(getattr(ds, "PatientAge", "")),
        "patient_weight": str(getattr(ds, "PatientWeight", "")),
        "patient_height": str(getattr(ds, "PatientSize", "")),
        "exam_date": _get_exam_date(ds),
        "exam_time": _get_exam_time(ds),
        "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
    }
    
    # Extract Waveform Data 
    if not hasattr(ds, "WaveformSequence"):
        raise ValueError(f"DICOM file {file_path} does not contain a WaveformSequence.")

    best_data = None
    best_signal_names = []
    max_samples = -1
    best_fs = None

    for wf in ds.WaveformSequence:
        waveform_data = getattr(wf, "WaveformData", None)
        if waveform_data is None and (0x5400, 0x1010) in wf:
            waveform_data = wf[0x5400, 0x1010].value
            
        if not waveform_data:
            continue

        num_channels = getattr(wf, "NumberOfWaveformChannels", 0)
        if num_channels == 0 and (0x003a, 0x0005) in wf:
            num_channels = wf[0x003a, 0x0005].value
            
        if num_channels <= 0:
            continue

        bits_allocated = getattr(wf, "WaveformBitsAllocated", 16)
        interpretation = str(getattr(wf, "WaveformSampleInterpretation", "SS")).strip().upper()
        signed = interpretation not in ("US", "UB")
        if bits_allocated == 8:
            dtype = np.int8 if signed else np.uint8
        elif bits_allocated == 32:
            dtype = np.int32 if signed else np.uint32
        else:
            dtype = np.int16 if signed else np.uint16

        data = np.frombuffer(waveform_data, dtype=dtype)
        
        remainder = len(data) % num_channels
        if remainder != 0:
            data = data[:-remainder] 
            
        try:
            data = data.reshape(-1, num_channels).T
        except ValueError:
            continue 

        # Keep the block with the longest duration
        if data.shape[1] > max_samples:
            max_samples = data.shape[1]
            best_data = data
            best_fs = getattr(wf, "SamplingFrequency", None)
            if best_fs is not None:
                try:
                    best_fs = float(best_fs)
                except (TypeError, ValueError):
                    best_fs = None
            
            signal_names = []
            channel_defs = getattr(wf, "ChannelDefinitionSequence", [])
            for i, ch_def in enumerate(channel_defs):
                name = f"lead_{i+1}"
                if hasattr(ch_def, "ChannelSourceSequence") and len(ch_def.ChannelSourceSequence) > 0:
                    if hasattr(ch_def.ChannelSourceSequence[0], "CodeMeaning"):
                        name = ch_def.ChannelSourceSequence[0].CodeMeaning
                elif hasattr(ch_def, "ChannelLabel"):
                    name = ch_def.ChannelLabel
                signal_names.append(name)
            best_signal_names = signal_names

    # Final check
    if best_data is None:
        raise ValueError("No valid continuous rhythm data could be extracted")

    # Apply to meta
    meta["fs"] = best_fs
    meta["n_samples"] = best_data.shape[1]
    meta["signal_names"] = best_signal_names
    signals = best_data

    return signals, meta

# HL7 aECG Reader
def read_hl7_aecg(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find tags
    def get_local_tag(tag):
        return tag.split('}')[-1] if '}' in tag else tag

    def find_child_by_local_name(parent, local_name):
        for child in parent:
            if get_local_tag(child.tag) == local_name:
                return child
        return None

    # Extract demographics
    patient_id = ""
    patient_firstname = ""
    patient_lastname = ""
    patient_birthdate = ""
    patient_age = ""
    gender = ""

    for trial_subject in root.iter():
        if get_local_tag(trial_subject.tag) == "trialSubject":
            id_node = find_child_by_local_name(trial_subject, "id")
            if id_node is not None:
                patient_id = id_node.get("extension", "")
            
            person_node = find_child_by_local_name(trial_subject, "subjectDemographicPerson")
            if person_node is not None:
                gender_node = find_child_by_local_name(person_node, "administrativeGenderCode")
                if gender_node is not None:
                    gender = gender_node.get("code", "")
                
                name_node = find_child_by_local_name(person_node, "name")
                if name_node is not None:
                    given_node = find_child_by_local_name(name_node, "given")
                    if given_node is not None and given_node.text:
                        patient_firstname = given_node.text.strip()
                    family_node = find_child_by_local_name(name_node, "family")
                    if family_node is not None and family_node.text:
                        patient_lastname = family_node.text.strip()
                
                # Check for birthTime
                birth_node = find_child_by_local_name(person_node, "birthTime")
                if birth_node is not None:
                    patient_birthdate = birth_node.get("value", "")
    
            break

    # Fallback:
    if not patient_id and not patient_lastname:
        for patient_node in root.iter():
            if get_local_tag(patient_node.tag) != "patient":
                continue
            id_node = find_child_by_local_name(patient_node, "id")
            if id_node is not None:
                patient_id = id_node.get("extension", "")
            person_node = find_child_by_local_name(patient_node, "patientPerson")
            if person_node is not None:
                gender_node = find_child_by_local_name(person_node, "administrativeGenderCode")
                if gender_node is not None:
                    gender = gender_node.get("code", "")
                name_node = find_child_by_local_name(person_node, "name")
                if name_node is not None:
                    given_node = find_child_by_local_name(name_node, "given")
                    if given_node is not None and given_node.text:
                        patient_firstname = given_node.text.strip()
                    family_node = find_child_by_local_name(name_node, "family")
                    if family_node is not None and family_node.text:
                        patient_lastname = family_node.text.strip()
                birth_node = find_child_by_local_name(person_node, "birthTime")
                if birth_node is not None:
                    patient_birthdate = birth_node.get("value", "")
                    # ECG-Format-Converter writes the age as <age value= unit="a">
                    # under birthTime when it has no birth date to give.
                    age_node = find_child_by_local_name(birth_node, "age")
                    if age_node is not None and age_node.get("value"):
                        unit = (age_node.get("unit") or "a").strip().lower()
                        patient_age = age_node.get("value").strip() + ("" if unit in ("a", "y", "yr", "year", "years") else unit)
            break

    # Find RHYTHM series
    target_series = None
    all_series = [elem for elem in root.iter() if get_local_tag(elem.tag) == "series"]
    
    for series in all_series:
        code_node = find_child_by_local_name(series, "code")
        if code_node is not None:
            code_val = code_node.get("code", "").upper()
            if "RHYTHM" in code_val:
                target_series = series
                break
    
    if target_series is None and len(all_series) > 0:
        target_series = all_series[0]

    if target_series is None:
        raise ValueError("No Waveform Series found in HL7 file.")

    # Extract time
    exam_date = ""
    exam_time = ""
    
    eff_time = find_child_by_local_name(target_series, "effectiveTime")
    if eff_time is not None:
        low_node = find_child_by_local_name(eff_time, "low")
        if low_node is not None:
            val = low_node.get("value", "")
            if len(val) >= 8: exam_date = val[:8]
            if len(val) >= 14: exam_time = val[8:14]
    
    waveform_component = None
    
    for child in target_series:
        tag = get_local_tag(child.tag)
        if tag == "component":
            if find_child_by_local_name(child, "sequenceSet") is not None:
                waveform_component = child
                break
    
    if waveform_component is None:
         raise ValueError("No valid waveform component found in RHYTHM series.")

    # Extract waveforms
    signals_dict = {}
    sampling_freq = None
    
    for seq in waveform_component.iter():
        if get_local_tag(seq.tag) != "sequence":
            continue

        code_node = find_child_by_local_name(seq, "code")
        if code_node is None: continue
        code_attr = code_node.get("code", "")
        
        # Time info
        if "TIME" in code_attr:
            val_node = find_child_by_local_name(seq, "value")
            if val_node is not None:
                inc_node = find_child_by_local_name(val_node, "increment")
                if inc_node is not None:
                    try:
                        val_str = inc_node.get("value", "1").replace(',', '.')
                        val = float(val_str)
                        unit = (inc_node.get("unit") or "s").strip().lower()
                        per_second = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
                                      "ms": 1e3, "msec": 1e3, "us": 1e6, "usec": 1e6,
                                      "\u00b5s": 1e6}.get(unit)
                        if val > 0 and per_second:
                            sampling_freq = per_second / val
                    except ValueError:
                        pass
        
        # Lead data
        elif "MDC_ECG_LEAD_" in code_attr:
            lead_name = code_attr.replace("MDC_ECG_LEAD_", "")
            val_node = find_child_by_local_name(seq, "value")
            if val_node is not None:
                digits_node = find_child_by_local_name(val_node, "digits")
                if digits_node is not None and digits_node.text:
                    clean_text = digits_node.text.replace('\n', ' ').strip()
                    try:
                        data = np.array([float(t) for t in clean_text.split()], dtype=float)

                        # Scale to uV
                        scale_node = find_child_by_local_name(val_node, "scale")
                        if scale_node is not None:
                            scale_val = scale_node.get("value", "1.0").replace(',', '.')
                            scale = float(scale_val)
                            unit = scale_node.get("unit", "uV")
                            if unit == "mV": scale *= 1000.0
                            data = data * scale
                        
                        signals_dict[lead_name] = data
                    except Exception:
                        pass

    if not signals_dict:
        raise ValueError("No waveform data found in the selected HL7 component.")

    # Map to FINAL_LEADS
    ordered_signals = []
    found_leads = []

    # Check for common keys
    for ref_lead in FINAL_LEADS:
        if ref_lead in signals_dict:
            ordered_signals.append(signals_dict[ref_lead])
            found_leads.append(ref_lead)
        else:
            for k in signals_dict.keys():
                if k.upper() == ref_lead.upper():
                    ordered_signals.append(signals_dict[k])
                    found_leads.append(ref_lead)
                    break
    
    if ordered_signals:
        collected, signal_names = ordered_signals, found_leads
    else:
        collected, signal_names = list(signals_dict.values()), list(signals_dict.keys())

    target_len = max(lead.size for lead in collected)
    signals = np.vstack([zero_pad(lead, target_len) for lead in collected])
    padded = {name: int(lead.size) for name, lead in zip(signal_names, collected) if lead.size < target_len}

    meta = {
        "source": "hl7",
        "fs": sampling_freq,
        "signal_names": signal_names,
        "n_samples": signals.shape[1],
        "patient_id": patient_id,
        "patient_firstname": patient_firstname,
        "patient_lastname": patient_lastname,
        "patient_birthdate": patient_birthdate,
        "patient_sex": gender,
        "patient_age": patient_age,
        "patient_weight": "",
        "patient_height": "",
        "exam_date": exam_date,
        "exam_time": exam_time,
        "pacemaker": "",
        "pon": None, 
        "poff": None, 
        "pdur": None
    }
    if padded:
        meta["padded_leads"] = padded
    return signals, meta

# XML Reader
def read_xml(file_path):
    # Check if file is HL7 aECG
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        if 'AnnotatedECG' in root.tag:
            try:
                return read_hl7_aecg(file_path)
            except Exception as e:
                raise ValueError(f"HL7 aECG parsing failed: {e}") from e
    except Exception as e:
        if "AnnotatedECG" in str(e) or "HL7" in str(e):
             raise e 
        pass

    # Standard XML Reader
    with open(file_path,'r',encoding='utf-8',errors='ignore') as f:
        xml_text = f.read()
    if "<lead " in xml_text and "ECG_RHYTHMS" not in xml_text:
        raise ValueError(f"{Path(file_path).name} looks like ECG-Format-Converter's generic XML "
                         "output (<ECG><lead name=...>), which cannot be read. Use the original "
                         "recording, or convert it to HL7 aECG, DICOM, WFDB or MAT instead.")
    tree = parse_xml(file_path)
    root = tree.getroot()

    patdata = root.find('.//patdata')
    patient_id = ""
    patient_firstname = ""
    patient_lastname = ""
    patient_birthdate = ""
    gender = ""
    weight = ""
    height = ""
    pacemaker = ""
    if patdata is not None:
        patient_id = patdata.findtext('id',default='').strip()
        patient_firstname = patdata.findtext('firstname',default='').strip()
        patient_lastname = patdata.findtext('lastname',default='').strip()
        patient_birthdate = patdata.findtext('birthdate',default='').strip()
        gender = patdata.findtext('gender',default='').strip().upper()
        for val_node in patdata.findall('.//value'):
            try:
                unit_node = val_node.getnext()
                if unit_node is not None and unit_node.text:
                    unit = unit_node.text.lower()
                    if 'kg' in unit: weight = val_node.text.strip()
                    elif 'cm' in unit: height = val_node.text.strip()
            except Exception:
                continue
        pacemaker = patdata.findtext('.//pacemaker/value',default='').strip().upper()

    exam_date = find_first_tag_text(root,'date')
    exam_time = find_first_tag_text(root,'time')

    rhythms_block = find_ecg_rhythms(root)
    if rhythms_block is None:
        snippet = extract_rhythms_snippet(xml_text)
        if snippet:
            if lxml_etree is None:
                raise ValueError("lxml is required to parse snippet but not available.")
            rhythms_block = lxml_etree.fromstring(snippet) # Use lxml for parsing snippet
        else:
            raise ValueError("Could not find ECG_RHYTHMS block in XML.")

    sampling_freq = detect_sampling_frequency(rhythms_block) or detect_sampling_frequency(root)

    lead_data={}
    signal_names = []
    channels = rhythms_block.findall('.//channel') + rhythms_block.findall('.//ECG_RHYTHM')
    for chan in channels:
        name=None
        for tag in ('name','NAME'):
            node=chan.find(tag)
            if node is not None and node.text:
                name=node.text.strip().upper(); break
        if not name:
            attr = chan.get('lead') or chan.get('name')
            if isinstance(attr,str):
                name=attr.strip().upper()
        if not name: continue
        data_node=None
        for tag in ('data','SAMPLES'):
            node=chan.find(tag)
            if node is not None and node.text:
                data_node=node; break
        if data_node is None: continue
        samples=parse_samples(data_node.text)
        if name not in lead_data:
            signal_names.append(name)
        lead_data[name]=samples

    if not lead_data:
        raise ValueError("No lead waveforms could be parsed from the ECG_RHYTHMS block.")
    target_len = max(lead_data[name].size for name in signal_names)
    padded = {name: int(lead_data[name].size) for name in signal_names
              if lead_data[name].size < target_len}
    signals = np.array([zero_pad(lead_data[name], target_len) for name in signal_names])

    meta = {
        "source": "xml",
        "fs": sampling_freq,
        "signal_names": signal_names,
        "n_samples": signals.shape[1],
        "patient_id": patient_id,
        "patient_firstname": patient_firstname,
        "patient_lastname": patient_lastname,
        "patient_birthdate": patient_birthdate,
        "patient_sex": gender,
        "patient_weight": weight,
        "patient_height": height,
        "exam_date": exam_date,
        "exam_time": exam_time,
        "pacemaker": pacemaker,
    }
    if padded:
        meta["padded_leads"] = padded
    return signals, meta
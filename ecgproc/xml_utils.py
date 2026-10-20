import re
import numpy as np

try:
    from lxml import etree as lxml_etree
    _xml_parser = lxml_etree.XMLParser(recover=True)
    def parse_xml(path): 
        return lxml_etree.parse(path, _xml_parser)
except ImportError:
    lxml_etree = None
    _xml_parser = None
    import xml.etree.ElementTree as std_etree
    def parse_xml(path): 
        return std_etree.parse(path)

def find_ecg_rhythms(root):
    for wd in root.findall('.//wavedata'):
        tnode = wd.find('type')
        if tnode is not None and tnode.text and tnode.text.strip().upper()=="ECG_RHYTHMS":
            return wd
        atype = wd.get('type')
        if atype and atype.strip().upper()=="ECG_RHYTHMS":
            return wd
    return None

def extract_rhythms_snippet(xml_text):
    pattern = r'(<wavedata[\s\S]*?<type>\s*ECG_RHYTHMS\s*</type>[\s\S]*?</wavedata>)'
    match = re.search(pattern, xml_text, flags=re.IGNORECASE)
    return match.group(1) if match else None

def parse_samples(raw_text):
    text = raw_text.strip()
    if ';' in text:
        tokens = [t.replace(',', '.') for t in re.split(r"[\s;]+", text) if t]
    else:
        tokens = [t for t in re.split(r"[\s,]+", text) if t]
    return np.array([float(t) for t in tokens], dtype=float)


def _rate(text):
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)", text or "")
    value = float(m.group(1).replace(',', '.')) if m else 0.0
    return value if 20 <= value <= 20000 else None

def find_first_tag_text(root, tagname):
    node = root.find(f'.//{tagname}')
    return node.text.strip() if node is not None and node.text else ''

def detect_sampling_frequency(root):
    sr_node = root.find('.//resolution/samplerate/value')
    if sr_node is not None and _rate(sr_node.text):
        return _rate(sr_node.text)
    for tag in ['samplerate','samplingrate','samplespersecond']:
        val = _rate(find_first_tag_text(root, tag))
        if val:
            return val
    chan = root.find('.//channel')
    if chan is None:
        chan = root.find('.//ECG_RHYTHM')
    if chan is not None:
        for attr in ['samplerate','samplingrate','samplespersecond']:
            val = _rate(chan.get(attr))
            if val:
                return val
    return None
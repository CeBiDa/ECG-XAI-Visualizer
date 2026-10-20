# ECG Processing Constants
FINAL_LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
LEAD_LABELS = {'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF'}


def lead_label(name):
    """Display form of a lead key: 'AVR' -> 'aVR', 'V1' -> 'V1'."""
    return LEAD_LABELS.get(str(name).upper(), str(name))


# Input formats
SUPPORTED_EXTENSIONS = [".mat", ".dat", ".hea", ".xml", ".hl7", ".csv", ".asc", ".dcm"]

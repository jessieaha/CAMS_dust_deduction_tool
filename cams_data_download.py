import cdsapi
import os 
import zipfile
import glob
VAR = "dust" #"dust" "particulate_matter_10um"
YEAR = 2024

# project_dir ='/tsn.tno.nl/Data/SV/sv-059025_unix/ProjectData/EU/CAMS/C71/Werkdocumenten/'
project_dir='.'
CAMS_folder_path =f'{project_dir}wp-dust/IRA_{VAR}/'
os.makedirs(CAMS_folder_path, exist_ok=True)

cams_downloads = [
    {"filename": f"CAMS_IRA_{(YEAR-1)}_12.zip", "year": [f"{(YEAR-1)}"], "month": ["12"]},
    {"filename": F"CAMS_IRA_{(YEAR)}_q1.zip", "year": [f"{YEAR}"], "month": ["01","02","03"]},
    {"filename": F"CAMS_IRA_{(YEAR)}_q2.zip", "year": [f"{YEAR}"], "month": ["04","05","06"]},
    {"filename": F"CAMS_IRA_{(YEAR)}_q3.zip", "year": [f"{YEAR}"], "month": ["07","08","09"]},
    {"filename": F"CAMS_IRA_{(YEAR)}_q4.zip", "year": [f"{YEAR}"], "month": ["10","11","12"]}
]

dataset = "cams-europe-air-quality-reanalyses"
client = cdsapi.Client()

for config in cams_downloads:
    filename = os.path.join(CAMS_folder_path, config["filename"])
    request = {
        "variable": [VAR],
        "model": ["ensemble"],
        "level": ["0"],
        "type": ["interim_reanalysis"],
        "year": config["year"],
        "month": config["month"]
    }
    
    client.retrieve(dataset, request).download(filename)


# Get list of all files in the current directory
files = glob.glob(f'{CAMS_folder_path}/*zip')

for file in files: 
   
    print(f"Extracting {file} to {CAMS_folder_path}...")
    
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(CAMS_folder_path)

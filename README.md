Full script for removing dust from EEA observation:
This notebook presents a prototype of the <u>**Dust Tool**</u> designed to support EU Member States in identifying and assessing natural contributions from dust to PM10 concentrations, as required by Article 16 of the Air Quality Directive (AAQD 2024/2881). The tool aims to subtract these natural dust contributions from reported PM10 concentrations to provide PM10 surface concentraton while removing natural sources.

> AAQD (2024/2881) Article 16: Member states are requested to identify a) zones where exceedances of limit values for a given pollutant are attributable to natural sources and b) average exposure territorial units, where exceedances of the level determined by the average exposure reduction obligations are attributable to natural sources.

Here, *the dust tool notebook product* supports the member states the European Commission DG-ENV had previously developed specific guidelines in 2011 (sec 2011-0208) , for the assessment of the natural contributions from dust, which need to be followed to allow subtraction of these contributions from the reported PM concentrations. The prototype focuses on daily data for Spain in 2024, demonstrating the methodology and applications.
# Data download 

You can either download from the webpage https://eeadmz1-downloads-webapp.azurewebsites.net/ or use API request in the script and load the data. 

For the metadata please download the new one through https://discomap.eea.europa.eu/App/AQViewer/index.html?fqn=Airquality_Dissem.b2g.measurements or use the 2024 one in the folder 

The Jupyternotebook can access util library share for all scrips 

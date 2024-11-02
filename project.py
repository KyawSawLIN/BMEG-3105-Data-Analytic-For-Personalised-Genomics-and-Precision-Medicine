from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
  
# fetch dataset 
heart_failure_clinical_records = fetch_ucirepo(id=519) 

  
# data (as pandas dataframes) 
X = heart_failure_clinical_records.data.features 
y = heart_failure_clinical_records.data.targets 
  
# metadata 
print(heart_failure_clinical_records.metadata) 
  
# variable information 
print(heart_failure_clinical_records.variables) 

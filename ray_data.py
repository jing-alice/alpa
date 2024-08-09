import json
import pandas as pd

with open('/root/ray_results/tune_fault_tolerance_cnn/trainable_219e7_00000_0_2024-03-22_16-35-26/result.json ') as f:
  data = json.load(f)
  
print("data: ",data)

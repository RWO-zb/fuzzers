import numpy as np
import pandas as pd

def get_rsd():
    file_name = '1.txt'
    res = []
    
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                
                name = parts[0]
        
                nums = [float(x) for x in parts[1:]]
                
               
                avg = np.mean(nums)
                std = np.std(nums, ddof=1)
                rsd = (std / avg) * 100 if avg != 0 else 0
                
                res.append({
                    "Model": name,
                    "Mean": round(avg, 2),
                    "Std_Dev": round(std, 2),
                    "RSD_Percent": round(rsd, 2)
                })
        
        df = pd.DataFrame(res)
        print(df.to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_rsd()
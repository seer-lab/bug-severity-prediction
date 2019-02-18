#data loading
import numpy as np
import pandas as pd
import math

def load_data(paths):
	# account for filename(s) as list or string
	if type(paths) is list:
		return [_load_data(path) for path in paths]
	else:
		return _load_data(paths)

def _load_data(path, ):
    df = pd.read_csv(path, sep=',', encoding='ISO-8859-1')
    raw_data = np.array(df)
    
    # get the columns for Subject and Severity Rating
    extract_cols = [1, 2]
    del_cols = np.delete(np.arange(raw_data.shape[1]), extract_cols)
    data = np.delete(raw_data, del_cols, axis=1)
    
    # check for possible NaN severity values
    del_rows = []
    for i in range(len(data)):
        if math.isnan(data[i][1]):
            del_rows.append(i)
    
    # delete rows that contain NaN severity values
    if len(del_rows) > 0:
        data = np.delete(data, del_rows, axis=0)
    
    return data
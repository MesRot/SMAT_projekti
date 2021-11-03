import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
import scipy.optimize as optimize

#read and create data
def ReadData():
    data = pd.read_excel("src\TACs.xlsx", sheet_name=None)
    relevant_keys = ['PT 300 ml Qclear 2', 'PT 300 ml Qclear 1', 'PT 200 ml Qclear 2', 'PT 200 ml Qclear 1', 'PT 100 ml Qclear 2', 'PT 100 ml Qclear 1']
    qclear = [2, 1, 2, 1, 2, 1]
    sizes = [300, 300, 200, 200, 100, 100]

    frames = []
    for key, clear, size in zip(relevant_keys, qclear, sizes):
        frame = data[key]
        df = frame.loc[5:]
        df.columns = ["time", "input", "tissue"]
        df[['time_start', 'time_end']] = df['time'].str.split(' - ', 1, expand=True)
        
        df = df.assign(sample=clear, flow=size)
        
        df = df.astype({'input': 'float64', 'tissue': 'float64', 'time_start': 'int32', 'time_end': 'int32'})
        df["midpoint"] = (df["time_end"] + df["time_start"]) / 2
        df = df.drop(["time"], axis=1)
        indx = np.argmax(df["input"] > max(df["input"]) * 0.1)
        df["rise_start"] = df["midpoint"].iloc[indx]
        frames.append(df)
    df = pd.concat(frames)
    return df

def CreateFeature_df(df):

    maxes = []
    sizes = []
    samples = []
    integrals = []
    colors = ["k", "b", "r", "c", "g", "m"]
    names = []
    input_maxes = []
    input_x_maxes = []
    tissue_x_maxes = []

    for name, group in df.groupby(['sample', 'flow']):
        sizes.append(group["flow"].iloc[0])
        q = group["sample"].iloc[1]
        if q == 1:
            samples.append(1)
        else:
            samples.append(2)
        cubic_tissue = interp1d(group["midpoint"], group["tissue"], kind='cubic')
        cubic_input = interp1d(group["midpoint"], group["input"], kind='cubic')

        min_inter, max_inter = min(group["midpoint"]), max(group["midpoint"])
        xnew = np.linspace(min_inter, max_inter, num=400, endpoint=True)
        #plt.title("Cubic splines")
        #plt.plot(group["midpoint"], group["input"], 'o', xnew, cubic_input(xnew), '-')
        
        integral = np.trapz(group["input"], x=group["midpoint"])
        integrals.append(integral)
        names.append(name)
        bounds = [(min_inter+1, max_inter-1)]
        
        results = optimize.minimize(lambda x: -cubic_input(x), x0=(100), bounds = bounds)
        input_x_maxes.append(results["x"][0])
        input_maxes.append(-results["fun"][0])
        
        results = optimize.minimize(lambda x: -cubic_tissue(x), x0=(100), bounds = bounds)
        tissue_x_maxes.append(results["x"][0])
        maxes.append(-results["fun"][0])

        indx = np.argmax(group["input"] > np.max(group["input"]) * 0.20)

    feature_df = pd.DataFrame({
        "flow": sizes,
        "tissue_max": maxes,
        "tissue_x_max": tissue_x_maxes,
        "sample": samples,
        "integral": integrals,
        "input_max": input_maxes,
        "input_x_max": input_x_maxes
    })
    feature_df["peak_time_difference"] = feature_df["tissue_x_max"] - feature_df["input_x_max"]
    feature_df
    df = pd.merge(df, feature_df, how="left", on=["flow", "sample"])
    return df
#end read and create data
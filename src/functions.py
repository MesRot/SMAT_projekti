import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


def EMG_loss(params, data): # NON PARAMETRIC
    errors = []
    groups = data.groupby(["flow", "sample"])
    for i, group in groups:
        mu = (group["input_x_max"] + params[0] * 1000 / group["flow"]).iloc[0]
        h = params[3]
        emg_vals = EMG(group["x"], mu, params[1], params[2], h)
        error = np.sum(((group["ss_tissue"] - emg_vals) ** 2) * group["error_weight"])
        errors.append(error)
    return sum(errors)

def EMG_loss_h(params, data, other_params): # NON PARAMETRIC
    errors = []
    groups = data.groupby(["flow", "sample"])
    for i, group in groups:
        mu = (group["input_x_max"] + other_params["mu_x"] * 1000 / group["flow"]).iloc[0]
        h = params[0]
        emg_vals = EMG(group["x"], mu, other_params["sigma"], other_params["lambda_"], h)
        #print(np.isnan(emg_vals).any())
        error = np.sum((group["s_tissue"] - emg_vals) ** 2)
        errors.append(error)
    return sum(errors)

def EMG(x, mu, sigma, lambda_, h):
    def formula1(x, mu, sigma):
        tau = 1 / lambda_
        z = (1 / np.sqrt(2)) * ((sigma / tau) - ((x - mu) / sigma))
        if z < 0:
            val = ((sigma) / tau) * np.sqrt(np.pi / 2) * np.exp(0.5 * (sigma / tau)**2 - ((x - mu) / tau)) * scipy.special.erfc(z)
        elif z < 6.71*10**7:
            val = np.exp(-0.5 * ((x - mu) / sigma)**2) * (sigma/tau) *  np.sqrt(np.pi / 2) * scipy.special.erfcx(z)
        else:
            val = np.exp(-0.5 * ((x - mu) / sigma)**2) / (1 + (((x  -mu) * tau) / sigma**2))
        return val
    y = np.asarray([formula1(x_, mu, sigma) for x_ in x])

    return h * y


def get_model_params(results_df, parameter_dict, flow=None):
    """
    parameter_dict sisältää parametrien määrän
    Esimerkiksi sigma_count kertoo sovitettavan polynomin astemäärän, 2 tarkoittaa toisen asteen polynomia
    josta tulee 3 parametria.

    returns: dictionary with exact values if flow is given, else returns the coefficients of the fit
    """ 
    #THIS function returns the value of evens in list

    def fit_2d_poly(x, y, poly_val, variable_to_predict=None):
        coefficients = np.polyfit(x, y, poly_val)
        if variable_to_predict:
            func = np.poly1d(coefficients)
            value = func(variable_to_predict)
            return value
        else:
            return coefficients
    
    parameter_dict = {
        "mu_x": fit_2d_poly(results_df["flow"], results_df["mu_x"], parameter_dict["mu_x_count"], variable_to_predict=flow),
        "sigma": fit_2d_poly(results_df["flow"], results_df["sigma"], parameter_dict["sigma_count"], variable_to_predict=flow),
        "lambda_": fit_2d_poly(results_df["flow"], results_df["lambda_"], parameter_dict["lambda_count"], variable_to_predict=flow),
        "h": fit_2d_poly(results_df["flow"], results_df["h"], parameter_dict["h_count"], variable_to_predict=flow),
    } 

    return parameter_dict

def get_selected_df(df, flow, sample):
    selected = df[df["flow"] == flow]
    selected = selected[selected["sample"] == sample]
    return selected


def get_interpolated_sample(start, stop, data, samples, variables, x_axis_var):
    x = np.linspace(start, stop, samples)
    samples = []
    for var in variables:
        spline = scipy.interpolate.interp1d(data[x_axis_var], data[var], kind='cubic')
        samples.append(spline(x))
    return x, samples

def plot_against_predictions(flow, sample, results_df, df, parameter_dict, save=False):
    #CREATE X-RANGE, GET MODEL PARAMETERS
    x_range = np.linspace(0, 350, 700)
    data = get_selected_df(df, flow, sample)
    input_integral = data["input_integral"].iloc[0]
    mu_x_dict, sigma_dict, lambda_dict, h_dict= get_model_params(flow, results_df, parameter_dict)
    # TRANSFORM MU_X -> MU

    #Otetaan arvot dictionaryista
    mu_x = mu_x_dict["mu_x"]
    sigma = sigma_dict["sigma"]
    lambda_ = lambda_dict["lambda_"]
    h_dict = h_dict["h"]

    mu = (data["input_x_max"] + mu_x * 1000 / data["flow"]).iloc[0]
    #CREATE Y-VALUES
    h = get_selected_df(results_df, flow, sample)["h"].iloc[0]
    emg_vals = EMG(x_range, mu, sigma, lambda_, h)
    plt.title(f"PARAMETRIC CURVE FIT, FLOW: {flow}, SAMPLE: {sample}")
    plt.plot(data["x"], data["s_tissue"], color="r")
    plt.plot(x_range, emg_vals, color="b")
    if save:
        plt.savefig(f"kuvaajat/param_curve_fit_{flow}_{sample}.png")
    plt.show()
    

def fit_curves(dataframe):
    groups = dataframe.groupby(["flow", "sample"]) # NON PARAMETRIC

    results_dict = {
        "flow": [],
        "sample": [],
        "mu": [],
        "mu_x": [],
        "lambda_": [],
        "sigma": [],
        "h": []
    }

    for i, group in groups:
        mu_x_alkuarvaus = 6.3259260
        sigma_alkuarvaus = 20
        lambda_alkuarvaus = 0.05

        bounds = [(None, None), (1e-5, None), (1e-5, None), (None, None)]

        x_0 = [mu_x_alkuarvaus, sigma_alkuarvaus, lambda_alkuarvaus, 1]
        print("STARTING FIRST OPTIMIZE")
        # SOVITTAA NORMALISOITUUN KÄYRÄÄN PARAMETRIT
        res1 = scipy.optimize.minimize(EMG_loss, args=group, x0=x_0, bounds=bounds)
        print(f"ENDING FIRST OPTIMIZE. PARAMS: {res1['x']}")
        #res = scipy.optimize.differential_evolution(EMG_loss, bounds=bounds, args=[group])
        x_range = np.linspace(0, 300, 900)
        mu = (group["input_x_max"] + res1["x"][0] * 1000 / group["flow"]).iloc[0]
        sigma = res1["x"][1]
        lambda_ = res1["x"][2]

        emg_params = {
            "mu_x": res1["x"][0],
            "sigma": sigma,
            "lambda_": lambda_
        }

        print("STARTING SECOND OPTIMIZE")
        # SOVITTAA KORKEUSPARAMETRIN NIIN ETTÄ SE NORMALISOIMATON MÄTSÄÄ OIKEAAN
        res2 = scipy.optimize.minimize(EMG_loss_h, args=(group, emg_params), x0=[80000])
        print(f"ENDING SECOND OPTIMIZE. PARAMS: {res2['x']}")

        h = res2["x"][0]


        pred_y = EMG(x_range, mu, sigma, lambda_, h)
        plt.plot(x_range, pred_y, color="b")
        plt.plot(group["x"], group["s_tissue"], color="r")
        #print(f'FLOW: {group["flow"].iloc[0]}, SAMPLE: {group["sample"].iloc[0]}, SIGMA: {res["x"][1]}, LAMBDA: {res["x"][2]}, MU_x: {res["x"][0]}')
        plt.show()
        results_dict["flow"].append(group["flow"].iloc[0])
        results_dict["sample"].append(group["sample"].iloc[0])
        results_dict["mu"].append(mu)
        results_dict["mu_x"].append(res1["x"][0])
        results_dict["sigma"].append(sigma)
        results_dict["lambda_"].append(lambda_)
        results_dict["h"].append(h)

    results_df = pd.DataFrame(results_dict)
    results_df = results_df.merge(dataframe, how="inner", on=["flow", "sample"])
    results_df

def get_r2(df, model_params):
    mu_x, sigma, lambda_, h = model_params 
    mu = (df["input_x_max"] + mu_x * 1000 / df["flow"]).iloc[0]
    y_vals = EMG(df["midpoint"], mu, sigma, lambda_, h)
    rss = np.sum((y_vals - df["tissue"]) ** 2)
    tss = np.sum((np.mean(df["tissue"]) - df["tissue"]) ** 2)
    return 1 - (rss / tss)

def plot_against_predictions_all(results_df, df, param_dict, save=False):
    x_range = np.linspace(0, 350, 700)
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    num_params = sum(param_dict.values()) + len(param_dict.values())


    fig.suptitle(f'Model predictions plotted against values with {num_params} parameters', fontsize=25)
    for i, group in df.groupby(["flow", "sample"]):
        flow, sample = i
        row, column = sample - 1, int(flow / 100 - 1)
        mu_x, sigma, lambda_, h = get_model_params(flow=flow, results_df = results_df, parameter_dict=param_dict)
        r2 = get_r2(group, [mu_x["mu_x"], sigma["sigma"], lambda_["lambda_"], h["h"]])
        r2 = str(round(r2, 2))
        mu = (group["input_x_max"] + mu_x["mu_x"] * 1000 / flow).iloc[0]
        emg_vals = EMG(x_range, mu, sigma["sigma"], lambda_["lambda_"], h["h"])
        ax[row, column].set_title(f"Flow: {flow}, Sample: {sample}, R2: {r2}", fontsize=15)
        ax[row, column].scatter(group["midpoint"], group["tissue"], color="r")
        ax[row, column].plot(x_range, emg_vals, color="b")
    if save:
        fig.savefig("all_predictions.jpg")
    plt.style.use("default")

def plot_all_params(results_df):
    plt.style.use("fivethirtyeight")
    fig, (ax1, ax2) = plt.subplots(2,2, sharex=True, figsize=(15, 12))
    fig.suptitle('Parameter plots against flow')
    fig.tight_layout(h_pad=5)

    ax1[0].scatter(results_df["flow"], results_df["mu_x"], color="red")
    ax1[0].set_title('mu_x')

    ax1[1].scatter(results_df["flow"], results_df["sigma"], color="orange")
    ax1[1].set_title('sigma')

    ax2[0].scatter(results_df["flow"], results_df["lambda_"], color="darkgreen")
    ax2[0].set_title('lambda')

    ax2[1].scatter(results_df["flow"], results_df["h"], color="blue")
    ax2[1].set_title('h')
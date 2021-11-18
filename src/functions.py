import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

def EMG_loss(params, data): # NON PARAMETRIC
    '''
    Calculates the error of the original normalized graph compared to the model graph.

    Original graph we are trying to model is first normalized to integrate to 1.
    Then the created EMG graph is compared to it (integrates to 1 too). Used norm is mean squared error.
    '''
    errors = []
    groups = data.groupby(["flow", "sample"])
    for i, group in groups:
        mu = (group["input_x_max"] + params[0] * 1000 / group["flow"]).iloc[0]
        h = params[3]
        emg_vals = EMG(group["x"], mu, params[1], params[2], h)
        error = np.mean(((group["ss_tissue"] - emg_vals) ** 2) * group["error_weight"])
        errors.append(error)
    return sum(errors)

def EMG_loss_h(params, data, other_params): # NON PARAMETRIC
    '''
    Calculates the error of the original (not normalized) graph compared to the model graph (scaled with custom h).

    Used norm is mean squared error.
    '''
    errors = []
    groups = data.groupby(["flow", "sample"])
    for i, group in groups:
        mu = (group["input_x_max"] + other_params["mu_x"] * 1000 / group["flow"]).iloc[0]
        h = params[0]
        emg_vals = EMG(group["x"], mu, other_params["sigma"], other_params["lambda_"], h)
        #print(np.isnan(emg_vals).any())
        error = np.mean((group["s_tissue"] - emg_vals) ** 2)
        errors.append(error)
    return sum(errors)
    
def EMG(x, mu, sigma, lambda_, h):
    '''
    Exponentially modified Gaussian distribution
    Returns the graph with specified parameters.

    mu is used to move peak of the graph
    sigma is used to control the width of the graph
    lambda_ is used to control the "tail" of the graph
    h is used for scaling the graph for right heigth, doesnt alter the shape

    The function is defined in the following way so that it behaves numerically better,
    see https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution for more information
    '''
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
    '''
    Returns the final parameters of the model; mu_x (not mu), sigma, lambda_ and h.
    
    Model parameters are based on optimized graphs created for each dataset. Each parameter has
    6 results for each dataset. Then a polynomial is fitted through the points resulting the final
    model parameter. 

    results_df = results of individual 
    parameter_dict = polynomial degree to be fitted through the points.
    flow = if flow is presented this returns parameter value for specified flow
        if flow is not presented this returns parameters of the model
    '''

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

    print(parameter_dict)

    return parameter_dict

def get_selected_df(df, flow, sample):
    '''
    Gets the desired part (what flow and what sample) of dataframe
    '''
    selected = df[df["flow"] == flow]
    selected = selected[selected["sample"] == sample]
    return selected

def get_interpolated_sample(start, stop, data, samples, variables, x_axis_var):
    '''
    Creates interpolated graph
    '''
    x = np.linspace(start, stop, samples)
    samples = []
    for var in variables:
        spline = scipy.interpolate.interp1d(data[x_axis_var], data[var], kind='cubic')
        samples.append(spline(x))
    return x, samples

def plot_against_predictions(flow, sample, results_df, df, parameter_dict, save=False):
    '''
    Plots the specified (same flow and sample) original graph and the modelled graph.
    '''
    #CREATE X-RANGE, GET MODEL PARAMETERS
    x_range = np.linspace(0, 350, 700)
    data = get_selected_df(df, flow, sample)

    input_integral = data["input_integral"].iloc[0]
    new_parameter_dict = get_model_params(results_df, parameter_dict, flow)
    # TRANSFORM MU_X -> MU

    # Get values from dictionary
    mu_x = new_parameter_dict["mu_x"]
    sigma = new_parameter_dict["sigma"]
    lambda_ = new_parameter_dict["lambda_"]
    h = new_parameter_dict["h"]

    mu = (data["input_x_max"] + mu_x * 1000 / data["flow"]).iloc[0]

    #CREATE Y-VALUES
    emg_vals = EMG(x_range, mu, sigma, lambda_, h)
    plt.title(f"PARAMETRIC CURVE FIT, FLOW: {flow}, SAMPLE: {sample}")
    plt.plot(data["x"], data["s_tissue"], color="r")
    plt.plot(x_range, emg_vals, color="b")
    if save:
        plt.savefig(f"kuvaajat/param_curve_fit_{flow}_{sample}.png")
    plt.show()

def fit_curves(dataframe):
    '''
    Optimization function.

    Finds optimized parameters for each dataset individually.
    '''
    groups = dataframe.groupby(["flow", "sample"]) # NON PARAMETRIC

    #Result parameters.
    results_dict = {
        "flow": [],
        "sample": [],
        "mu": [],
        "mu_x": [],
        "lambda_": [],
        "sigma": [],
        "h": []
    }

    #Optimize for every dataset.
    for i, group in groups:
        # Guesses for the parameters, used in optimization.
        mu_x_initial_guess = 6.3259260
        sigma_initial_guess = 20
        lambda_initial_guess = 0.05

        bounds = [(None, None), (1e-5, None), (1e-5, None), (None, None)]

        x_0 = [mu_x_initial_guess, sigma_initial_guess, lambda_initial_guess, 1]
        print("STARTING FIRST OPTIMIZE")

        # Fits a graph to original NORMALIZED graph here.
        # Finds optimal parameters for SPECIFIED DATASET ONLY.
        res1 = scipy.optimize.minimize(EMG_loss, args=group, x0=x_0, bounds=bounds)
        print(f"ENDING FIRST OPTIMIZE. PARAMS: {res1['x']}")
        #res = scipy.optimize.differential_evolution(EMG_loss, bounds=bounds, args=[group])
        x_range = np.linspace(0, 300, 900)
        mu = (group["input_x_max"] + res1["x"][0] * 1000 / group["flow"]).iloc[0]
        sigma = res1["x"][1]
        lambda_ = res1["x"][2]

        # Parameters for the optimized graph for the specified dataset.
        emg_params = {
            "mu_x": res1["x"][0],
            "sigma": sigma,
            "lambda_": lambda_
        }

        print("STARTING SECOND OPTIMIZE")
        # Finds the heigh scaling parameter to scale the modelled graph to original graph
        res2 = scipy.optimize.minimize(EMG_loss_h, args=(group, emg_params), x0=[80000])
        print(f"ENDING SECOND OPTIMIZE. PARAMS: {res2['x']}")

        #Scale parameter h for the specified dataset.
        h = res2["x"][0]

        # Plot the results for visualization.
        # Append the results dictionary with results.
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

    # Final results.
    # results_df contains all the optimized parameters (mu_x, sigma, lambda_, h) for individual datasets.
    results_df = pd.DataFrame(results_dict)
    results_df = results_df.merge(dataframe, how="inner", on=["flow", "sample"])
    return results_df

def get_r2(df, model_params):
    '''
    Returns the r-squared of the model.
    '''
    mu_x, sigma, lambda_, h = model_params 
    mu = (df["input_x_max"] + mu_x * 1000 / df["flow"]).iloc[0]
    y_vals = EMG(df["midpoint"], mu, sigma, lambda_, h)
    rss = np.sum((y_vals - df["tissue"]) ** 2)
    tss = np.sum((np.mean(df["tissue"]) - df["tissue"]) ** 2)
    return 1 - (rss / tss)

def plot_against_predictions_all(results_df, df, param_dict, save=False):
    '''
    Plots all orignal graph and their respective modelled graphs.
    '''
    x_range = np.linspace(0, 350, 700)
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    num_params = sum(param_dict.values()) + len(param_dict.values())


    fig.suptitle(f'Model predictions plotted against values with {num_params} parameters', fontsize=25)
    for i, group in df.groupby(["flow", "sample"]):
        flow, sample = i
        row, column = sample - 1, int(flow / 100 - 1)
        new_parameter_dict = get_model_params(flow=flow, results_df = results_df, parameter_dict=param_dict)
        r2 = get_r2(group, [new_parameter_dict["mu_x"], new_parameter_dict["sigma"], new_parameter_dict["lambda_"], new_parameter_dict["h"]])
        r2 = str(round(r2, 2))
        mu = (group["input_x_max"] + new_parameter_dict["mu_x"] * 1000 / flow).iloc[0]
        emg_vals = EMG(x_range, mu, new_parameter_dict["sigma"], new_parameter_dict["lambda_"], new_parameter_dict["h"])
        ax[row, column].set_title(f"Flow: {flow}, Sample: {sample}, R2: {r2}", fontsize=15)
        ax[row, column].scatter(group["midpoint"], group["tissue"], color="r")
        ax[row, column].plot(x_range, emg_vals, color="b")
    if save:
        fig.savefig("all_predictions.jpg")
    plt.style.use("default")

def plot_all_params(results_df, parameter_dict):
    '''
    Plots all parameters and the polynomial fitted through them.
    The fitted polynomial represents the final model parameter.
    '''
    plt.style.use("fivethirtyeight")
    fig, (ax1, ax2) = plt.subplots(2,2, sharex=True, figsize=(15, 12))
    fig.suptitle('Parameter plots against flow')
    fig.tight_layout(h_pad=5)

    #Plot all parameter values to own figure.
    ax1[0].scatter(results_df["flow"], results_df["mu_x"], color="red")
    ax1[0].set_title('mu_x')

    ax1[1].scatter(results_df["flow"], results_df["sigma"], color="magenta")
    ax1[1].set_title('sigma')

    ax2[0].scatter(results_df["flow"], results_df["lambda_"], color="darkgreen")
    ax2[0].set_title('lambda')

    ax2[1].scatter(results_df["flow"], results_df["h"], color="blue")
    ax2[1].set_title('h')

    if parameter_dict:
        # Gets a dictionary of all final model parameters.
        coefficients = get_model_params(results_df, parameter_dict, None)

        #Add fitted polynomial to the figures.
        func = np.poly1d(coefficients["mu_x"])
        x_range = np.linspace(100, 300)
        y_values = func(x_range)

        ax1[0].plot(x_range, y_values)

        func = np.poly1d(coefficients["sigma"])
        y_values = func(x_range)

        ax1[1].plot(x_range, y_values)

        func = np.poly1d(coefficients["lambda_"])
        y_values = func(x_range)

        ax2[0].plot(x_range, y_values)

        func = np.poly1d(coefficients["h"])
        y_values = func(x_range)

        ax2[1].plot(x_range, y_values)
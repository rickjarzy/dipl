fitted_products_list = ["double_fft_sg_15",
                        "fft_sg_15",
                        "lin_poly_win15.weights.1_0.5_0.01_0.01",
                        "lin_poly_win15.weights.1_0.5_0.01_0.01_q0.01",
                        "poly_15.1_0.01_0.01_0.01",
                        "poly_lin_win15.weights.1_0.5_0.01_0.01",
                        "sg_15_fft_poly",
                        "single_fft_sg_15",
                        "fft",
                        ]

fit_info_all = {}
fit_info_poly = {}
fit_info_fft = {}
fit_info_best = {}
doy_factors = {}
# All fits together in one dict
# ----------------------------------------------------------------------------------------------------------------------
#fit_info_all["double_fft_sg_15"] = {"algo_desc": "calculates twice a FFT using a SQ window of 15 days", "weights":"no", "files_list":[]}
fit_info_all["single_fft_sg_15"] = {"algo_desc": "calculates a single FFT over a SG window of 15 days", "weights":"no", "files_list":[]}
#fit_info_all["fft_sg_15"] = {"algo_desc": " calculates a FFT over a SG window of 15 days", "weights":"no", "files_list":[]}
fit_info_all["sg_15_fft_poly"] = {"algo_desc":"calculates a FFT first and then calculates a Poly Fit twice in a SG window of 15 days", "weights":"yes", "files_list":[]}

fit_info_all["fft"] = {"algo_desc":"calculates a single FFT over an entire year - 46 epochs - recalculates the values via FFT and writes out each file of the year new", "weights":"no", "files_list":[]}


# i have only 3 fitted epochs - i think this is because they have exact the same results as poly_lin_win15.
# fit_info_all["lin_poly_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
#                                                                              "updateing the weights", "weights":"yes", "files_list":[]}

# fitted products start with 2000113 for band_1 and band_2
fit_info_all["lin_poly_win15.weights.1_0.5_0.01_0.01_q0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
                                                                             "updateing the weights", "weights":"yes", "files_list":[]}
# fitted products start with 2000113 for band_1 and band_2
fit_info_all["poly_15.1_0.01_0.01_0.01"] = {"algo_desc": "calculates a poly fit with a SG window of 15 and no lin interpolation", "weights":"yes", "files_list":[]}

fit_info_all["poly_lin_win15.weights.1_0.5_0.25_0.01"] = {"algo_desc": "calculates a poly fit with a SG window of 15 and no lin interpolation", "weights":"yes", "files_list":[]}
# fitted products start with 2000113 for band_1 and band_2
fit_info_all["poly_lin_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc":"calculates a double poly fit with a SG window first and then a lin interpolation"
                                                                "on the pixels that had NaN values or less then a specific inversion number", "files_list":[]}
fit_info_all["dft.elements_3.1.00_0.50_0.25_0.01"] = {"algo_desc":"calculates a discrete fourier transformation over the entire year ", "weights":"yes", "files_list":[]}
# FFT FITs Only in this dict
# ----------------------------------------------------------------------------------------------------------------------
#fit_info_fft["double_fft_sg_15"] = {"algo_desc": "calculates twice a FFT using a SQ window of 15 days", "weights":"no", "files_list":[]}
fit_info_fft["single_fft_sg_15"] = {"algo_desc": "calculates a single FFT over a SG window of 15 days", "weights":"no", "files_list":[]}
fit_info_fft["sg_15_fft_poly"] = {"algo_desc":"calculates a FFT first and then calculates a Poly Fit twice in a SG window of 15 days", "weights":"yes", "files_list":[]}
fit_info_fft["fft"] = {"algo_desc":"calculates a single FFT over an entire year - 46 epochs - recalculates the values via FFT and writes out each file of the year new", "weights":"no", "files_list":[]}

fit_info_fft["dft.elements_3.1.00_0.50_0.25_0.01"] = {"algo_desc":"calculates a discrete fourier transformation over the entire year ", "weights":"yes", "files_list":[]}

# POLY FITs Only in this dict
# ----------------------------------------------------------------------------------------------------------------------
# lin_poly_win15.weights.1_0.5_0.01_0.01 hat aktuell nur 3 epochen - weil ich glaub es ergaben sich di total gleichen werte wie in poly_lin
# fit_info_poly["lin_poly_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
#                                                                              "updateing the weights", "weights":"yes", "files_list":[]}

fit_info_poly["lin_poly_win15.weights.1_0.5_0.01_0.01_q0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
                                                                             "updateing the weights", "weights":"yes", "files_list":[]}

fit_info_poly["poly_15.1_0.01_0.01_0.01"] = {"algo_desc": "calculates a poly fit with a SG window of 15 and no lin interpolation", "weights":"yes", "files_list":[]}
fit_info_poly["poly_lin_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc":"calculates a double poly fit with a SG window first and then a lin interpolation"
                                                                "on the pixels that had NaN values or less then a specific inversion number", "files_list":[]}
fit_info_all["poly_lin_win15.weights.1_0.5_0.25_0.01"] = {"algo_desc": "calculates a poly fit with a SG window of 15 and no lin interpolation", "weights":"yes", "files_list":[]}


# BEST FITS
# ----------------------------------------------------------------------------------------------------------------------
fit_info_best["fft"] = {"algo_desc":"calculates a single FFT over an entire year - 46 epochs - recalculates the values via FFT and writes out each file of the year new", "weights":"no", "files_list":[]}
fit_info_best["dft.elements_3.1.00_0.50_0.25_0.01"] = {"algo_desc": "calculate discrete fourier transform with three elements for entire year"}
fit_info_best["poly_lin_win15.weights.1_0.5_0.25_0.01"] = {"algo_desc": "calculates a poly fit with a SG window of 15 and no lin interpolation", "weights":"yes", "files_list":[]}
fit_info_best["poly_lin_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc":"calculates a double poly fit with a SG window first and then a lin interpolation"}
# create the factor for the list index calculation in the main programm of the MODISPlots3.py
# Start 2001 has factor 0
# I dont have the full year of 2000 because the first epochs start with doy 57, so the plots start with year 2001
for year in range(1,21,1):
    doy_factors[2000+year] = {"factor":year-1}


if __name__ == "__main__":

    print("\nImport Fit Information Successful")
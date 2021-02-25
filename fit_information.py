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

fit_info_all["double_fft_sg_15"] = {"algo_desc": "calculates twice a FFT using a SQ window of 15 days", "weights":"no", "files_list":[]}
fit_info_all["single_fft_sg_15"] = {"algo_desc": "calculates a single FFT over a SG window of 15 days", "weights":"no", "files_list":[]}
fit_info_all["fft_sg_15"] = {"algo_desc": " calculates a FFT over a SG window of 15 days", "weights":"no", "files_list":[]}
fit_info_all["sg_15_fft_poly"] = {"algo_desc":"calculates a FFT first and then calculates a Poly Fit twice in a SG window of 15 days", "weights":"yes", "files_list":[]}

fit_info_all["fft"] = {"algo_desc":"calculates a single FFT over an entire year - 46 epochs - recalculates the values via FFT and writes out each file of the year new", "weights":"no", "files_list":[]}



fit_info_all["lin_poly_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
                                                                             "updateing the weights", "weights":"yes", "files_list":[]}
fit_info_all["lin_poly_win15.weights.1_0.5_0.01_0.01_q0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
                                                                             "updateing the weights", "weights":"yes", "files_list":[]}
fit_info_all["poly_15.1_0.01_0.01_0.01"] = {"algo_desc": "calculates a poly fit with a SG window of 15 and no lin interpolation", "weights":"yes", "files_list":[]}
fit_info_all["poly_lin_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc":"calculates a double poly fit with a SG window first and then a lin interpolation"
                                                                "on the pixels that had NaN values or less then a specific inversion number", "files_list":[]}


fit_info_fft["double_fft_sg_15"] = {"algo_desc": "calculates twice a FFT using a SQ window of 15 days", "weights":"no", "files_list":[]}
fit_info_fft["single_fft_sg_15"] = {"algo_desc": "calculates a single FFT over a SG window of 15 days", "weights":"no", "files_list":[]}
fit_info_fft["fft_sg_15"] = {"algo_desc": " calculates a FFT over a SG window of 15 days", "weights":"no", "files_list":[]}
fit_info_fft["sg_15_fft_poly"] = {"algo_desc":"calculates a FFT first and then calculates a Poly Fit twice in a SG window of 15 days", "weights":"yes", "files_list":[]}

fit_info_fft["fft"] = {"algo_desc":"calculates a single FFT over an entire year - 46 epochs - recalculates the values via FFT and writes out each file of the year new", "weights":"no", "files_list":[]}


fit_info_poly["lin_poly_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
                                                                             "updateing the weights", "weights":"yes", "files_list":[]}
fit_info_poly["lin_poly_win15.weights.1_0.5_0.01_0.01_q0.01"] = {"algo_desc": "calculates first a linear regresion on all NaN Pixels and then perfomrs twice a Poly Fit with "
                                                                             "updateing the weights", "weights":"yes", "files_list":[]}
fit_info_poly["poly_15.1_0.01_0.01_0.01"] = {"algo_desc": "calculates a poly fit with a SG window of 15 and no lin interpolation", "weights":"yes", "files_list":[]}
fit_info_poly["poly_lin_win15.weights.1_0.5_0.01_0.01"] = {"algo_desc":"calculates a double poly fit with a SG window first and then a lin interpolation"
                                                                "on the pixels that had NaN values or less then a specific inversion number", "files_list":[]}

if __name__ == "__main__":

    print("\nImport Fit Information Successful")
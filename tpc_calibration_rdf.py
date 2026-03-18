import argparse
import json
import os
import sys

import numpy as np
import ROOT
import uproot
import yaml

sys.path.append("utils")
import utils as utils


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(1111)
ROOT.ROOT.EnableImplicitMT()


P_BINS_DEFAULT = np.linspace(0.0, 2.5, 50)
FUNC_STRING = (
    "([1] - TMath::Power((TMath::Abs(2*x/2.80839160743) / "
    "TMath::Sqrt(1 + 2*2*x*x/2.80839160743/2.80839160743)),[3]) - "
    "TMath::Log([2] + TMath::Power(1/TMath::Abs(2*x/2.80839160743),[4]))) "
    "* [0] / TMath::Power((TMath::Abs(2*x/2.80839160743) / "
    "TMath::Sqrt(1 + 2*2*x*x/2.80839160743/2.80839160743)),[3])"
)
DATA_BB_PARAMS = (-321.34, 0.6539, 1.591, 0.8225, 2.363)
MC_BB_PARAMS = (-611.592275784047, 1.7160058505736953, 3.2070341629338754, -0.7453309037015425, 4.107037208529488)


def infer_data_tree_name(input_file_name):
    tree_names = ["O2datahypcands", "O2hypcands", "O2hypcandsflow"]
    tree_keys = uproot.open(input_file_name).keys()
    for tree in tree_names:
        for key in tree_keys:
            if tree in key:
                return tree
    raise RuntimeError(f"Could not determine input tree name from data file: {input_file_name}")


def build_selection_string(selection_dict, matter):
    selection_string = utils.convert_sel_to_rdf_string(selection_dict) if selection_dict else "true"
    if matter == "matter":
        selection_string += " && fIsMatter == true"
    elif matter == "antimatter":
        selection_string += " && fIsMatter == false"
    return selection_string


def get_default_selection():
    return {
        "fAvgClusterSizeHe": "fAvgClusterSizeHe > 6",
        "fNSigmaHe3": "fNSigmaHe3 > -3",
        "fNTPCclusHe": "fNTPCclusHe > 90",
        "fTPCmomHe": "abs(fTPCmomHe) > 0.5 && abs(fTPCmomHe) < 5.0",
    }


def make_bb_function(name, parameters, color):
    func = ROOT.TF1(name, FUNC_STRING, 0.5, 6.0, 5)
    for i_par, value in enumerate(parameters):
        func.SetParameter(i_par, value)
    func.SetLineColor(color)
    return func


def build_rdf(file_names, tree_name, calibrate_he_momentum, is_mc, is_h4l):
    rdf = ROOT.RDataFrame(utils.build_chain(file_names, tree_name))
    rdf = utils.correct_and_convert_rdf(
        rdf,
        calibrate_he3_pt=calibrate_he_momentum,
        isMC=is_mc,
        isH4L=is_h4l,
    )
    if is_mc:
        column_names = {str(col) for col in rdf.GetColumnNames()}
        if "fIsReco" in column_names:
            rdf = rdf.Filter("fIsReco == true")
    return rdf


def calibrate_sample(sample_name, rdf, selection_string, output_file, p_bins, dedx_bins, dedx_range, default_bb_params):
    sample_rdf = rdf.Filter(selection_string).Define("fTPCmomHeAbs", "abs(fTPCmomHe)")
    p_bins_array = np.asarray(p_bins, dtype=np.float64)
    n_p_bins = len(p_bins) - 1

    is_data = sample_name == "data"

    sample_dir = output_file.mkdir(sample_name)
    sample_dir.cd()

    hist_2d = sample_rdf.Histo2D(
        (
            f"hTPCdEdXvsP_{sample_name}",
            r";#it{p}/z (GeV/#it{c}); d#it{E}/d#it{X} (a. u.)",
            n_p_bins,
            p_bins_array,
            dedx_bins,
            dedx_range[0],
            dedx_range[1],
        ),
        "fTPCmomHeAbs",
        "fTPCsignalHe",
    ).GetValue()

    hist_to_fit = ROOT.TH1D(
        f"hTPCdEdXvsP_toFit_{sample_name}",
        r";#it{p}/z (GeV/#it{c}); d#it{E}/d#it{X} (a. u.)",
        n_p_bins,
        p_bins_array,
    )

    slices_dir = sample_dir.mkdir("TPCdEdX_slices")

    for i_p in range(n_p_bins):
        p_low = p_bins[i_p]
        p_high = p_bins[i_p + 1]
        p_label = f"{p_low:.2f} #leq #it{{p}}/z < {p_high:.2f} GeV/#it{{c}}"
        hist_slice = hist_2d.ProjectionY(
            f"hTPCdEdX_{sample_name}_p{i_p}",
            i_p + 1,
            i_p + 1,
            "e",
        )
        hist_slice.SetTitle(p_label + r";d#it{E}/d#it{X} (a. u.);counts")
        hist_slice.SetDirectory(0)

        if hist_slice.GetEntries() == 0:
            continue

        mean = hist_slice.GetMean()
        rms = hist_slice.GetRMS()
        fit_min = max(dedx_range[0], mean - 4.0 * rms)
        fit_max = min(dedx_range[1], mean + 4.0 * rms)

        if is_data:
            # Double Gaussian: signal (peaked at higher dE/dx) + background (peaked at lower dE/dx)
            # [0]*gaus(x,[1],[2]) + [3]*gaus(x,[4],[5])
            # signal: [0]=amp_sig, [1]=mean_sig, [2]=sigma_sig
            # background: [3]=amp_bkg, [4]=mean_bkg, [5]=sigma_bkg
            double_gaus = ROOT.TF1(
                f"double_gaus_{sample_name}_p{i_p}",
                "gaus(0) + gaus(3)",
                fit_min,
                fit_max,
            )
            max_val = hist_slice.GetMaximum()
            # Signal Gaussian: near the mean
            double_gaus.SetParameter(0, max_val)
            double_gaus.SetParameter(1, mean)
            double_gaus.SetParameter(2, rms * 0.5)
            # Background Gaussian: peaked at lower values
            double_gaus.SetParameter(3, max_val * 0.1)
            double_gaus.SetParameter(4, mean - 1.5 * rms)
            double_gaus.SetParameter(5, rms)

            # Constrain signal amplitude > 0, sigma > 0
            double_gaus.SetParLimits(0, 0, max_val * 10)
            double_gaus.SetParLimits(2, 0, rms * 5)
            # Constrain background amplitude > 0, sigma > 0, mean < signal mean
            double_gaus.SetParLimits(3, 0, max_val * 10)
            double_gaus.SetParLimits(4, fit_min, mean)
            double_gaus.SetParLimits(5, 0, rms * 5)

            double_gaus.SetParName(0, "amp_sig")
            double_gaus.SetParName(1, "mean_sig")
            double_gaus.SetParName(2, "sigma_sig")
            double_gaus.SetParName(3, "amp_bkg")
            double_gaus.SetParName(4, "mean_bkg")
            double_gaus.SetParName(5, "sigma_bkg")

            # First pass: fit to get signal mean estimate
            hist_slice.Fit(double_gaus, "MQRSL0", "", fit_min, fit_max)
            sig_mean_est = double_gaus.GetParameter(1)
            bkg_mean_est = double_gaus.GetParameter(4)

            # If background mean ended up above signal mean, swap and refit
            if bkg_mean_est > sig_mean_est:
                double_gaus.SetParameter(1, bkg_mean_est)
                double_gaus.SetParameter(4, sig_mean_est)

            # Enforce mean_bkg < mean_sig via a combined function with shared constraint
            # Use a custom function: par[4] = mean_sig - delta, delta > 0
            double_gaus_constrained = ROOT.TF1(
                f"double_gaus_cstr_{sample_name}_p{i_p}",
                "[0]*TMath::Gaus(x,[1],[2]) + [3]*TMath::Gaus(x,[1]-[4],[5])",
                fit_min,
                fit_max,
            )
            double_gaus_constrained.SetParName(0, "amp_sig")
            double_gaus_constrained.SetParName(1, "mean_sig")
            double_gaus_constrained.SetParName(2, "sigma_sig")
            double_gaus_constrained.SetParName(3, "amp_bkg")
            double_gaus_constrained.SetParName(4, "delta_mean")
            double_gaus_constrained.SetParName(5, "sigma_bkg")

            sig_mean_init = double_gaus.GetParameter(1)
            bkg_mean_init = double_gaus.GetParameter(4)
            delta_init = max(sig_mean_init - bkg_mean_init, 0.1 * rms)

            double_gaus_constrained.SetParameter(0, double_gaus.GetParameter(0))
            double_gaus_constrained.SetParameter(1, sig_mean_init)
            double_gaus_constrained.SetParameter(2, double_gaus.GetParameter(2))
            double_gaus_constrained.SetParameter(3, double_gaus.GetParameter(3))
            double_gaus_constrained.SetParameter(4, delta_init)
            double_gaus_constrained.SetParameter(5, double_gaus.GetParameter(5))

            double_gaus_constrained.SetParLimits(0, 0, max_val * 10)
            double_gaus_constrained.SetParLimits(2, 0, rms * 5)
            double_gaus_constrained.SetParLimits(3, 0, max_val * 10)
            double_gaus_constrained.SetParLimits(4, 0, mean - fit_min)  # delta >= 0 ensures bkg mean < sig mean
            double_gaus_constrained.SetParLimits(5, 0, rms * 5)

            fit_status = hist_slice.Fit(double_gaus_constrained, "MQRSL+", "", fit_min, fit_max)
            # Use the signal Gaussian mean and sigma for the BB fit
            hist_to_fit.SetBinContent(i_p + 1, double_gaus_constrained.GetParameter(1))
            hist_to_fit.SetBinError(i_p + 1, double_gaus_constrained.GetParameter(2))
        else:
            fit_status = hist_slice.Fit("gaus", "MQRSL+", "", fit_min, fit_max)
            fit_func = hist_slice.GetFunction("gaus")
            hist_to_fit.SetBinContent(i_p + 1, fit_func.GetParameter(1))
            hist_to_fit.SetBinError(i_p + 1, fit_func.GetParameter(2))

        slices_dir.cd()
        hist_slice.Write()
        sample_dir.cd()

    func_default_data = make_bb_function(f"func_BB_default_data_{sample_name}", DATA_BB_PARAMS, ROOT.kRed + 1)
    func_default_mc = make_bb_function(f"func_BB_default_mc_{sample_name}", MC_BB_PARAMS, ROOT.kGreen + 2)
    func_fit = make_bb_function(f"func_BB_fit_{sample_name}", default_bb_params, ROOT.kBlue + 1)
    fit_result = hist_to_fit.Fit(func_fit, "MQRS")

    canvas = ROOT.TCanvas(f"cTPCdEdXvsP_{sample_name}", f"cTPCdEdXvsP_{sample_name}", 800, 600)
    canvas.DrawFrame(p_bins[0], dedx_range[0], p_bins[-1], dedx_range[1], r";#it{p}/z (GeV/#it{c}); d#it{E}/d#it{X} (a. u.)")
    hist_2d.Draw("colz same")
    func_default_data.Draw("L same")
    func_default_mc.Draw("L same")
    func_fit.Draw("L same")

    legend = ROOT.TLegend(0.15, 0.72, 0.45, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(func_default_data, "Default BB data", "l")
    legend.AddEntry(func_default_mc, "Default BB MC", "l")
    legend.AddEntry(func_fit, "Fitted BB", "l")
    legend.Draw()

    hist_to_fit.GetListOfFunctions().Clear()
    cv_hist = ROOT.TCanvas(f"cTPCdEdXvsP_fit_{sample_name}", f"cTPCdEdXvsP_fit_{sample_name}", 800, 600)
    cv_hist.DrawFrame(p_bins[0], dedx_range[0], p_bins[-1], dedx_range[1], r";#it{p}/z (GeV/#it{c}); d#it{E}/d#it{X} (a. u.)")
    hist_to_fit.SetLineColor(ROOT.kBlack)
    hist_to_fit.SetMarkerStyle(20)
    hist_to_fit.SetMarkerColor(ROOT.kBlack)
    hist_to_fit.Draw("E")
    func_default_data.Draw("L same")
    func_default_mc.Draw("L same")
    func_fit.Draw("L same")
    legend_hist = ROOT.TLegend(0.15, 0.68, 0.45, 0.88)
    legend_hist.SetBorderSize(0)
    legend_hist.SetFillStyle(0)
    legend_hist.AddEntry(hist_to_fit, "Gaussian means", "lep")
    legend_hist.AddEntry(func_default_data, "Default BB data", "l")
    legend_hist.AddEntry(func_default_mc, "Default BB MC", "l")
    legend_hist.AddEntry(func_fit, "Fitted BB", "l")
    legend_hist.Draw()

    sample_dir.cd()
    hist_2d.Write()
    hist_to_fit.Write()
    func_default_data.Write()
    func_default_mc.Write()
    func_fit.Write()
    cv_hist.Write()
    canvas.Write()

    return {
        "sample": sample_name,
        "default_parameters": [float(val) for val in default_bb_params],
        "fit_parameters": [float(func_fit.GetParameter(i)) for i in range(func_fit.GetNpar())],
        "fit_parameter_errors": [float(func_fit.GetParError(i)) for i in range(func_fit.GetNpar())],
        "fit_status": int(fit_result.Status()),
        "chi2": float(func_fit.GetChisquare()),
        "ndf": int(func_fit.GetNDF()),
    }


def main():
    parser = argparse.ArgumentParser(description="TPC calibration with ROOT RDataFrame for data and MC.")
    parser.add_argument("--config-file", dest="config_file", default="", help="Path to the YAML file with configuration.")
    args = parser.parse_args()

    if args.config_file == "":
        print("** No config file provided. Exiting. **")
        raise SystemExit(1)

    with open(args.config_file, "r", encoding="utf-8") as config_file:
        config = yaml.full_load(config_file)

    output_dir = config["output_dir"]
    output_file_name = config.get("output_file", "tpc_calibration_rdf.root")
    os.makedirs(output_dir, exist_ok=True)

    selection = config.get("selection", get_default_selection())
    matter = config.get("is_matter", "both")
    calibrate_he_momentum = config.get("calibrate_he_momentum", True)
    p_bins = np.asarray(config.get("p_bins", P_BINS_DEFAULT), dtype=np.float64)
    dedx_bins = int(config.get("dedx_bins", 175))
    dedx_range = config.get("dedx_range", [0.0, 2000.0])

    selection_string = build_selection_string(selection, matter)

    samples = []

    input_files_data = config.get("input_files_data", [])
    if input_files_data:
        data_tree_name = infer_data_tree_name(input_files_data[0])
        data_rdf = build_rdf(
            input_files_data,
            data_tree_name,
            calibrate_he_momentum=calibrate_he_momentum,
            is_mc=False,
            is_h4l=True,
        )
        samples.append(("data", data_rdf, DATA_BB_PARAMS))

    input_files_mc_h3l = config.get("input_files_mc_h3l", [])
    if input_files_mc_h3l:
        mc_h3l_rdf = build_rdf(
            input_files_mc_h3l,
            "O2mchypcands",
            calibrate_he_momentum=calibrate_he_momentum,
            is_mc=True,
            is_h4l=False,
        )
        samples.append(("mc_h3l", mc_h3l_rdf, MC_BB_PARAMS))

    input_files_mc_h4l = config.get("input_files_mc_h4l", [])
    if input_files_mc_h4l:
        mc_h4l_rdf = build_rdf(
            input_files_mc_h4l,
            "O2mchypcands",
            calibrate_he_momentum=calibrate_he_momentum,
            is_mc=True,
            is_h4l=True,
        )
        samples.append(("mc_h4l", mc_h4l_rdf, MC_BB_PARAMS))

    if not samples:
        raise RuntimeError("No input samples configured. Please provide at least one data or MC input list.")

    output_path = os.path.join(output_dir, output_file_name)
    params_output_path = os.path.splitext(output_path)[0] + "_params.json"
    output_file = ROOT.TFile.Open(output_path, "RECREATE")
    fit_summary = {}

    for sample_name, rdf, default_bb_params in samples:
        fit_summary[sample_name] = calibrate_sample(
            sample_name,
            rdf,
            selection_string,
            output_file,
            p_bins,
            dedx_bins,
            dedx_range,
            default_bb_params,
        )

    output_file.Close()
    with open(params_output_path, "w", encoding="utf-8") as params_file:
        json.dump(fit_summary, params_file, indent=2)
    print(f"Output written to {output_path}")
    print(f"Fit parameters written to {params_output_path}")


if __name__ == "__main__":
    main()

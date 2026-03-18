import argparse
import os

import numpy as np
import ROOT
import uproot
import yaml


import sys
sys.path.append('utils')
import utils as utils

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
ROOT.ROOT.EnableImplicitMT()

kOrangeC = ROOT.TColor.GetColor("#ff7f00")

def main():
    parser = argparse.ArgumentParser(description="Configure the parameters of the script.")
    parser.add_argument("--config-file", dest="config_file", default="", help="Path to the YAML file with configuration.")
    args = parser.parse_args()
    if args.config_file == "":
        print("** No config file provided. Exiting. **")
        raise SystemExit(1)

    with open(args.config_file, "r", encoding="utf-8") as config_file:
        config = yaml.full_load(config_file)

    input_file_name_data = config["input_files_data"]
    input_file_name_mc_h3l = config["input_files_mc_h3l"]
    input_file_name_mc_h4l = config["input_files_mc_h4l"]
    output_dir = config["output_dir"]
    output_file_name = config["output_file"]
    colliding_system = config["colliding_system"]
    selections = config["selection"]
    pid_selections = config.get("pid_selection", {})
    is_matter = config["is_matter"]
    calibrate_he_momentum = config["calibrate_he_momentum"]

    os.makedirs(output_dir, exist_ok=True)

    # Base selection (no PID) — used for wrong mass templates
    selections_string_no_pid = utils.convert_sel_to_rdf_string(selections) if selections else "true"
    # Full selection (with PID) — used for data, signal MC
    all_selections = {**selections, **pid_selections}
    selections_string = utils.convert_sel_to_rdf_string(all_selections) if all_selections else "true"

    if is_matter == "matter":
        selections_string += " && fIsMatter == true"
        selections_string_no_pid += " && fIsMatter == true"
    elif is_matter == "antimatter":
        selections_string += " && fIsMatter == false"
        selections_string_no_pid += " && fIsMatter == false"

    tree_names = ["O2datahypcands", "O2hypcands", "O2hypcandsflow"]
    tree_keys = uproot.open(input_file_name_data[0]).keys()
    tree_name = None
    for tree in tree_names:
        for key in tree_keys:
            if tree in key:
                tree_name = tree
                break
        if tree_name is not None:
            break
    if tree_name is None:
        raise RuntimeError("Could not determine input tree name from data file")


    he3_spectrum = ROOT.TF1('mtexpo', '[2]*x*exp(-TMath::Sqrt([0]*[0]+x*x)/[1])', 0.1, 6)
    he3_spectrum.FixParameter(0, 2.99131)
    he3_spectrum.FixParameter(1, 0.5199)
    he3_spectrum.FixParameter(2, 1.0)
    he4_spectrum = he3_spectrum.Clone('he4_spectrum')
    he4_spectrum.FixParameter(0, 3.72738)


    data_rdf = utils.correct_and_convert_rdf(
        ROOT.RDataFrame(utils.build_chain(input_file_name_data, tree_name)),
        calibrate_he3_pt=calibrate_he_momentum,
        isMC=False,
    )
    mc_rdf_h3l_full = utils.correct_and_convert_rdf(
        ROOT.RDataFrame(utils.build_chain(input_file_name_mc_h3l, "O2mchypcands")),
        calibrate_he3_pt=calibrate_he_momentum,
        isMC=True,
        pt_spectrum=he3_spectrum,
    )
    mc_rdf_h4l_full = utils.correct_and_convert_rdf(
        ROOT.RDataFrame(utils.build_chain(input_file_name_mc_h4l, "O2mchypcands")),
        calibrate_he3_pt=calibrate_he_momentum,
        isMC=True,
        pt_spectrum=he4_spectrum,
    )

    print("------------------------")
    print("Data loaded and converted to RDF. Starting analysis...")
    print("Rejection flag added to MC RDFs. Starting to filter RDFs with selections...")
    mc_rdf_h3l_base = mc_rdf_h3l_full.Filter("rej == 1 && fIsReco == true")
    mc_rdf_h4l_base = mc_rdf_h4l_full.Filter("rej == 1 && fIsReco == true")
    data_rdf = data_rdf.Filter(selections_string)
    mc_rdf_h3l = mc_rdf_h3l_base.Filter(selections_string)
    mc_rdf_h4l = mc_rdf_h4l_base.Filter(selections_string)

    print("-------------------------")
    print("Data filtered with selections. Starting to create histograms...")

    th1_pt_h3l = mc_rdf_h3l.Histo1D(("pt_h3l_mc", "pt_h3l_mc", 100, 0, 10), "fAbsGenPt")
    th1_pt_h4l = mc_rdf_h4l.Histo1D(("pt_h4l_mc", "pt_h4l_mc", 100, 0, 10), "fAbsGenPt")
    th2_n_sigma_he3_pt_mc = mc_rdf_h3l.Histo2D(("n_sigma_he3_pt_mc", "n_sigma_he3_pt_mc", 100, 0, 10, 100, -5, 5), "fAbsGenPt", "fNSigmaHe3")
    th2_n_sigma_he4_pt_mc = mc_rdf_h4l.Histo2D(("n_sigma_he4_pt_mc", "n_sigma_he4_pt_mc", 100, 0, 10, 100, -5, 5), "fAbsGenPt", "fNSigmaHe4")
    th2_n_sigma_he3_pt_data = data_rdf.Histo2D(("n_sigma_he3_pt_data", "n_sigma_he3_pt_data", 100, 0, 10, 100, -5, 5), "fPt", "fNSigmaHe3")
    th2_n_sigma_he4_pt_data = data_rdf.Histo2D(("n_sigma_he4_pt_data", "n_sigma_he4_pt_data", 100, 0, 10, 100, -5, 5), "fPt", "fNSigmaHe4")
    th2_wrong_mass_h3l_pt_mc = mc_rdf_h3l.Histo2D(("wrong_mass_h3l_pt_mc", "wrong_mass_h3l_pt_mc", 100, 0, 10, 100, 3.89, 3.97), "fAbsGenPt", "fMassH4L")
    th2_wrong_mass_h4l_pt_mc = mc_rdf_h4l.Histo2D(("wrong_mass_h4l_pt_mc", "wrong_mass_h4l_pt_mc", 100, 0, 10, 100, 2.96, 3.04), "fAbsGenPt", "fMassH3L")
    th2_dedx_tpcmom = data_rdf.Histo2D(("dedx_tpcmom_data", "dedx_tpcmom_data", 100, 0, 5, 1000, 0, 2000), "fTPCmomHe", "fTPCsignalHe")
    th2_dedx_tpcmom_mc_h3l = mc_rdf_h3l.Histo2D(("dedx_tpcmom_mc_h3l", "dedx_tpcmom_mc_h3l", 1000, 0, 5, 100, 0, 2000), "fTPCmomHe", "fTPCsignalHe")
    th2_dedx_tpcmom_mc_h4l = mc_rdf_h4l.Histo2D(("dedx_tpcmom_mc_h4l", "dedx_tpcmom_mc_h4l", 1000, 0, 5, 100, 0, 2000), "fTPCmomHe", "fTPCsignalHe")

    print("-------------------------")
    print("Histograms created. Starting signal extraction...")

    mass_window_sel = "fMassH4L > 3.89 && fMassH4L < 3.97 && fMassH3L > 2.96 && fMassH3L < 3.04"
    data_rdf = data_rdf.Filter(mass_window_sel)
    mc_rdf_h3l = mc_rdf_h3l.Filter(mass_window_sel)
    mc_rdf_h4l = mc_rdf_h4l.Filter(mass_window_sel)

    # Wrong mass templates: no PID selection for maximum statistics
    mc_rdf_h3l_no_pid = mc_rdf_h3l_base.Filter(selections_string_no_pid).Filter(mass_window_sel)
    mc_rdf_h4l_no_pid = mc_rdf_h4l_base.Filter(selections_string_no_pid).Filter(mass_window_sel)

    inv_mass_string_h4l = "#it{M}_{^{4}He+#pi^{-}}" if is_matter == "matter" else "#it{M}_{^{4}#bar{He}+#pi^{+}}"
    inv_mass_string_h3l = "#it{M}_{^{3}He+#pi^{-}}" if is_matter == "matter" else "#it{M}_{^{3}#bar{He}+#pi^{+}}"

    mass3_min = data_rdf.Min("fMassH3L").GetValue()
    mass3_max = data_rdf.Max("fMassH3L").GetValue()
    mass4_min = data_rdf.Min("fMassH4L").GetValue()
    mass4_max = data_rdf.Max("fMassH4L").GetValue()
    mass3HL = ROOT.RooRealVar("mass3HL", inv_mass_string_h3l, mass3_min, mass3_max, "GeV/c^{2}")
    mass4HL = ROOT.RooRealVar("mass4HL", inv_mass_string_h4l, mass4_min, mass4_max, "GeV/c^{2}")

    mass_roo_mc_h3l = utils.rdf_to_roodataset(mc_rdf_h3l, "fMassH3L", mass3HL, "histo_mc_h3l")
    mass_roo_mc_h4l = utils.rdf_to_roodataset(mc_rdf_h4l, "fMassH4L", mass4HL, "histo_mc_h4l")

    ## first extract h3l and h4l templates from MC with a double sided crystal ball
    mu3HL = ROOT.RooRealVar("mu_h3l", "hypernucl mass", mass3_min, mass3_max, "GeV/c^{2}")
    sigma_h3l = ROOT.RooRealVar("sigma_h3l", "hypernucl width", 0.001, 0.0024, "GeV/c^{2}")
    a1_h3l = ROOT.RooRealVar("a1_h3l", "a1_h3l", 0., 5.)
    a2_h3l = ROOT.RooRealVar("a2_h3l", "a2_h3l", 0., 5.)
    n1_h3l = ROOT.RooRealVar("n1_h3l", "n1_h3l", 0., 5.)
    n2_h3l = ROOT.RooRealVar("n2_h3l", "n2_h3l", 0., 5.)
    pars_h3l = [mu3HL, sigma_h3l, a1_h3l, a2_h3l, n1_h3l, n2_h3l]
    signal_h3l = ROOT.RooCrystalBall("cb_h3l", "cb_h3l_cl", mass3HL, mu3HL, sigma_h3l, a1_h3l, n1_h3l, a2_h3l, n2_h3l)
    signal_h3l.fitTo(mass_roo_mc_h3l)
    frame_h3l = mass3HL.frame()
    frame_h3l.SetName("frame_h3l_mc")
    mass_roo_mc_h3l.plotOn(frame_h3l)
    signal_h3l.plotOn(frame_h3l)

    mu4HL = ROOT.RooRealVar("mu_h4l", "hypernucl mass", 3.91, 3.95, "GeV/c^{2}")
    sigma_h4l = ROOT.RooRealVar("sigma_h4l", "hypernucl width", 0.001, 0.004, "GeV/c^{2}")
    a1_h4l = ROOT.RooRealVar("a1_h4l", "a1_h4l", 0., 5.)
    a2_h4l = ROOT.RooRealVar("a2_h4l", "a2_h4l", 0., 5.)
    n1_h4l = ROOT.RooRealVar("n1_h4l", "n1_h4l", 0., 5.)
    n2_h4l = ROOT.RooRealVar("n2_h4l", "n2_h4l", 0., 5.)
    pars_h4l = [mu4HL, sigma_h4l, a1_h4l, a2_h4l, n1_h4l, n2_h4l]
    signal_h4l = ROOT.RooCrystalBall("cb_h4l", "cb_h4l_cl", mass4HL, mu4HL, sigma_h4l, a1_h4l, n1_h4l, a2_h4l, n2_h4l)
    signal_h4l.fitTo(mass_roo_mc_h4l)
    frame_h4l = mass4HL.frame()
    frame_h4l.SetName("frame_h4l_mc")
    mass_roo_mc_h4l.plotOn(frame_h4l)
    signal_h4l.plotOn(frame_h4l)

    ## fix all the params except for the mu
    for par in pars_h3l:
        if par.GetName() == "mu_h3l":
            continue
        if par.GetName() == "sigma_h3l":
            par.setRange(par.getVal(), par.getVal() * 1.2)
            continue
        par.setConstant(True)
    for par in pars_h4l:
        if par.GetName() == "mu_h4l":
            continue
        if par.GetName() == "sigma_h4l":
            par.setRange(par.getVal(), par.getVal() * 1.2)
            continue
        par.setConstant(True)

    ### now get the mc template of h4l for real h3l candidates and vice versa
    mass3HL.setBins(30)
    mass4HL.setBins(30)

    mass_roo_mc_h3l_wrong_mass = utils.rdf_to_roodataset(mc_rdf_h4l_no_pid, "fMassH3L", mass3HL, "histo_mc_h3l_wrong_mass")
    mass_roo_mc_h4l_wrong_mass = utils.rdf_to_roodataset(mc_rdf_h3l_no_pid, "fMassH4L", mass4HL, "histo_mc_h4l_wrong_mass")
    # Binned templates
    datahist_h3l_wrong_mass = ROOT.RooDataHist("dh_h3l_wrong_mass", "dh_h3l_wrong_mass", ROOT.RooArgList(mass3HL), mass_roo_mc_h3l_wrong_mass)
    datahist_h4l_wrong_mass = ROOT.RooDataHist("dh_h4l_wrong_mass", "dh_h4l_wrong_mass", ROOT.RooArgList(mass4HL), mass_roo_mc_h4l_wrong_mass)
    histpdf_h3l_wrong_mass = ROOT.RooHistPdf("histpdf_h3l_wrong_mass", "histpdf_h3l_wrong_mass", ROOT.RooArgSet(mass3HL), datahist_h3l_wrong_mass, 0)
    histpdf_h4l_wrong_mass = ROOT.RooHistPdf("histpdf_h4l_wrong_mass", "histpdf_h4l_wrong_mass", ROOT.RooArgSet(mass4HL), datahist_h4l_wrong_mass, 0)
    # Gaussian kernel for smoothing via FFT convolution
    smooth_mean = ROOT.RooRealVar("smooth_mean", "smooth_mean", 0.0)
    smooth_width_h3l = ROOT.RooRealVar("smooth_width_h3l", "smooth_width_h3l", 0.003)  # tune as needed
    smooth_width_h4l = ROOT.RooRealVar("smooth_width_h4l", "smooth_width_h4l", 0.003)
    gauss_kernel_h3l = ROOT.RooGaussian("gauss_kernel_h3l", "gauss_kernel_h3l", mass3HL, smooth_mean, smooth_width_h3l)
    gauss_kernel_h4l = ROOT.RooGaussian("gauss_kernel_h4l", "gauss_kernel_h4l", mass4HL, smooth_mean, smooth_width_h4l)
    # FFT convolution: smoothed histogram PDF
    mass3HL.setBins(10000, "cache")
    mass4HL.setBins(10000, "cache")
    pdf_h3l_wrong_mass = ROOT.RooFFTConvPdf("mc_pdf_h3l_wrong_mass", "mc_pdf_h3l_wrong_mass", mass3HL, histpdf_h3l_wrong_mass, gauss_kernel_h3l)
    pdf_h4l_wrong_mass = ROOT.RooFFTConvPdf("mc_pdf_h4l_wrong_mass", "mc_pdf_h4l_wrong_mass", mass4HL, histpdf_h4l_wrong_mass, gauss_kernel_h4l)
    pdf_h3l_wrong_mass.setBufferFraction(0.2)
    pdf_h4l_wrong_mass.setBufferFraction(0.2)

    # Compare histogram vs smoothed PDF
    frame_h3l_wrong_mass = mass3HL.frame()
    frame_h3l_wrong_mass.SetName("frame_h3l_wrong_mass")
    mass_roo_mc_h3l_wrong_mass.plotOn(frame_h3l_wrong_mass)
    pdf_h3l_wrong_mass.plotOn(frame_h3l_wrong_mass, ROOT.RooFit.LineColor(ROOT.kRed))
    frame_h4l_wrong_mass = mass4HL.frame()
    frame_h4l_wrong_mass.SetName("frame_h4l_wrong_mass")
    mass_roo_mc_h4l_wrong_mass.plotOn(frame_h4l_wrong_mass)
    pdf_h4l_wrong_mass.plotOn(frame_h4l_wrong_mass, ROOT.RooFit.LineColor(ROOT.kRed))

    c0_bkg_h3l = ROOT.RooRealVar("c0_bkg_h3l", "c0_bkg_h3l", -1, 1.0)
    c1_bkg_h3l = ROOT.RooRealVar("c1_bkg_h3l", "c1_bkg_h3l", -1, 1.0)
    bkg_h3l = ROOT.RooChebychev("bkg_h3l", "bkg_h3l", mass3HL, ROOT.RooArgList(c0_bkg_h3l, c1_bkg_h3l))
    c0_bkg_h4l = ROOT.RooRealVar("c0_bkg_h4l", "c0_bkg_h4l", -1, 1.0)
    c1_bkg_h4l = ROOT.RooRealVar("c1_bkg_h4l", "c1_bkg_h4l", -1, 1.0)
    bkg_h4l = ROOT.RooChebychev("bkg_h4l", "bkg_h4l", mass4HL, ROOT.RooArgList(c0_bkg_h4l, c1_bkg_h4l))

    nsig_h3l = ROOT.RooRealVar("nsig_h3l", "signal events", 100, 0, 1e5)
    n_bkg = ROOT.RooRealVar("nbkg", "background events", 100, 0, 1e5)
    n_sig_h4l = ROOT.RooRealVar("nsig_h4l", "signal events", 100, 0, 1e5)
    model_h3l = ROOT.RooAddPdf("model_h3l", "model_h3l", ROOT.RooArgList(signal_h3l, pdf_h3l_wrong_mass, bkg_h3l), ROOT.RooArgList(nsig_h3l, n_sig_h4l, n_bkg))
    model_h4l = ROOT.RooAddPdf("model_h4l", "model_h4l", ROOT.RooArgList(signal_h4l, pdf_h4l_wrong_mass, bkg_h4l), ROOT.RooArgList(n_sig_h4l, nsig_h3l, n_bkg))
    mass_roo_data_h3l = utils.rdf_to_roodataset(data_rdf, "fMassH3L", mass3HL, "histo_data_h3l")
    mass_roo_data_h4l = utils.rdf_to_roodataset(data_rdf, "fMassH4L", mass4HL, "histo_data_h4l")

    categories = ROOT.RooCategory("categories", "categories")
    categories.defineType("h3l")
    categories.defineType("h4l")
    data = ROOT.RooDataSet(
        "data",
        "data",
        ROOT.RooArgSet(mass3HL, mass4HL, categories),
        ROOT.RooFit.Index(categories),
        ROOT.RooFit.Import("h3l", mass_roo_data_h3l),
        ROOT.RooFit.Import("h4l", mass_roo_data_h4l),
    )
    roosim = ROOT.RooSimultaneous("roosim", "roosim", categories)
    roosim.addPdf(model_h3l, "h3l")
    roosim.addPdf(model_h4l, "h4l")
    roosim.fitTo(data, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True))

    mass3HL.setRange("signal", mu3HL.getVal() - 3 * sigma_h3l.getVal(), mu3HL.getVal() + 3 * sigma_h3l.getVal())
    mass4HL.setRange("signal", mu4HL.getVal() - 3 * sigma_h4l.getVal(), mu4HL.getVal() + 3 * sigma_h4l.getVal())
    signal_h3l_int = signal_h3l.createIntegral(ROOT.RooArgSet(mass3HL), ROOT.RooArgSet(mass3HL), "signal")
    signal_h3l_int_val_3s = signal_h3l_int.getVal() * nsig_h3l.getVal()
    signal_h3l_int_val_3s_error = signal_h3l_int_val_3s * nsig_h3l.getError() / nsig_h3l.getVal()
    signal_h4l_int = signal_h4l.createIntegral(ROOT.RooArgSet(mass4HL), ROOT.RooArgSet(mass4HL), "signal")
    signal_h4l_int_val_3s = signal_h4l_int.getVal() * n_sig_h4l.getVal()
    signal_h4l_int_val_3s_error = signal_h4l_int_val_3s * n_sig_h4l.getError() / n_sig_h4l.getVal()

    bkg_int_h3l = bkg_h3l.createIntegral(ROOT.RooArgSet(mass3HL), ROOT.RooArgSet(mass3HL))
    bkg_int_h3l_val_3s = bkg_int_h3l.getVal() * n_bkg.getVal()
    bkg_int_h3l_val_3s_error = bkg_int_h3l_val_3s * n_bkg.getError() / n_bkg.getVal()
    wrong_mass_h3l_int = pdf_h3l_wrong_mass.createIntegral(ROOT.RooArgSet(mass3HL), ROOT.RooArgSet(mass3HL))
    wrong_mass_h3l_int_val_3s = wrong_mass_h3l_int.getVal() * n_sig_h4l.getVal()
    wrong_mass_h3l_int_val_3s_error = wrong_mass_h3l_int_val_3s * n_sig_h4l.getError() / n_sig_h4l.getVal()
    s_b_ratio_h3l = signal_h3l_int_val_3s / (bkg_int_h3l_val_3s + wrong_mass_h3l_int_val_3s)
    s_b_ratio_h3l_error = s_b_ratio_h3l * np.sqrt(
        (signal_h3l_int_val_3s_error / signal_h3l_int_val_3s) ** 2
        + (bkg_int_h3l_val_3s_error / bkg_int_h3l_val_3s) ** 2
        + (wrong_mass_h3l_int_val_3s_error / wrong_mass_h3l_int_val_3s) ** 2
    )

    bkg_int_h4l = bkg_h4l.createIntegral(ROOT.RooArgSet(mass4HL), ROOT.RooArgSet(mass4HL))
    bkg_int_h4l_val_3s = bkg_int_h4l.getVal() * n_bkg.getVal()
    bkg_int_h4l_val_3s_error = bkg_int_h4l_val_3s * n_bkg.getError() / n_bkg.getVal()
    wrong_mass_h4l_int = pdf_h4l_wrong_mass.createIntegral(ROOT.RooArgSet(mass4HL), ROOT.RooArgSet(mass4HL))
    wrong_mass_h4l_int_val_3s = wrong_mass_h4l_int.getVal() * nsig_h3l.getVal()
    wrong_mass_h4l_int_val_3s_error = wrong_mass_h4l_int_val_3s * nsig_h3l.getError() / nsig_h3l.getVal()
    s_b_ratio_h4l = signal_h4l_int_val_3s / (bkg_int_h4l_val_3s + wrong_mass_h4l_int_val_3s)
    s_b_ratio_h4l_error = s_b_ratio_h4l * np.sqrt(
        (signal_h4l_int_val_3s_error / signal_h4l_int_val_3s) ** 2
        + (bkg_int_h4l_val_3s_error / bkg_int_h4l_val_3s) ** 2
        + (wrong_mass_h4l_int_val_3s_error / wrong_mass_h4l_int_val_3s) ** 2
    )

    pinfo_h3l = ROOT.TPaveText(0.632, 0.5, 0.932, 0.85, "NDC")
    pinfo_h3l.SetBorderSize(0)
    pinfo_h3l.SetFillStyle(0)
    pinfo_h3l.SetTextAlign(11)
    pinfo_h3l.SetTextFont(42)
    pinfo_h3l.AddText(f"Signal (S): {signal_h3l_int_val_3s:.0f} #pm {signal_h3l_int_val_3s_error:.0f}")
    pinfo_h3l.AddText(f"S/B (3 #sigma): {s_b_ratio_h3l:.1f} #pm {s_b_ratio_h3l_error:.1f}")
    pinfo_h3l.AddText("#mu = " + f"{mu3HL.getVal() * 1e3:.2f} #pm {mu3HL.getError() * 1e3:.2f}" + " MeV/#it{c}^{2}")
    pinfo_h3l.AddText("#sigma = " + f"{sigma_h3l.getVal() * 1e3:.2f} #pm {sigma_h3l.getError() * 1e3:.2f}" + " MeV/#it{c}^{2}")

    pinfo_h4l = ROOT.TPaveText(0.632, 0.5, 0.932, 0.85, "NDC")
    pinfo_h4l.SetBorderSize(0)
    pinfo_h4l.SetFillStyle(0)
    pinfo_h4l.SetTextAlign(11)
    pinfo_h4l.SetTextFont(42)
    pinfo_h4l.AddText(f"Signal (S): {signal_h4l_int_val_3s:.0f} #pm {signal_h4l_int_val_3s_error:.0f}")
    pinfo_h4l.AddText(f"S/B (3 #sigma): {s_b_ratio_h4l:.1f} #pm {s_b_ratio_h4l_error:.1f}")
    pinfo_h4l.AddText("#mu = " + f"{mu4HL.getVal() * 1e3:.2f} #pm {mu4HL.getError() * 1e3:.2f}" + " MeV/#it{c}^{2}")
    pinfo_h4l.AddText("#sigma = " + f"{sigma_h4l.getVal() * 1e3:.2f} #pm {sigma_h4l.getError() * 1e3:.2f}" + " MeV/#it{c}^{2}")

    eff_3hl_mc = mc_rdf_h3l.Count().GetValue() / mc_rdf_h3l_full.Count().GetValue()
    eff_4hl_mc = mc_rdf_h4l.Count().GetValue() / mc_rdf_h4l_full.Count().GetValue()
    br_h3l = 0.25
    br_h4l = 0.55

    frame_data_h3l = mass3HL.frame()
    frame_data_h3l.SetName("frame_data_h3l")
    mass_roo_data_h3l.plotOn(frame_data_h3l)
    model_h3l.plotOn(frame_data_h3l)
    model_h3l.plotOn(frame_data_h3l, ROOT.RooFit.Components("cb_h3l"), ROOT.RooFit.LineColor(kOrangeC), ROOT.RooFit.LineStyle(2))
    model_h3l.plotOn(frame_data_h3l, ROOT.RooFit.Components("mc_pdf_h3l_wrong_mass"), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.LineStyle(2))
    model_h3l.plotOn(frame_data_h3l, ROOT.RooFit.Components("bkg_h3l"), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(2))
    frame_data_h3l.addObject(pinfo_h3l)

    frame_data_h4l = mass4HL.frame()
    frame_data_h4l.SetName("frame_data_h4l")
    mass_roo_data_h4l.plotOn(frame_data_h4l)
    model_h4l.plotOn(frame_data_h4l)
    model_h4l.plotOn(frame_data_h4l, ROOT.RooFit.Components("cb_h4l"), ROOT.RooFit.LineColor(kOrangeC), ROOT.RooFit.LineStyle(2))
    model_h4l.plotOn(frame_data_h4l, ROOT.RooFit.Components("mc_pdf_h4l_wrong_mass"), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.LineStyle(2))
    model_h4l.plotOn(frame_data_h4l, ROOT.RooFit.Components("bkg_h4l"), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(2))
    frame_data_h4l.addObject(pinfo_h4l)

    print("-------------------------")
    print("Signal extraction completed. Printing results and saving histograms...")
    print("efficiency h3l: ", eff_3hl_mc)
    print("efficiency h4l: ", eff_4hl_mc)
    print("nsig_h3l: ", nsig_h3l.getVal(), nsig_h3l.getError())
    print("nsig_h4l: ", n_sig_h4l.getVal(), n_sig_h4l.getError())
    print("h3l / h4l: ", nsig_h3l.getVal() / n_sig_h4l.getVal() / (eff_3hl_mc / eff_4hl_mc) / (br_h3l / br_h4l))


    tfile = ROOT.TFile.Open(os.path.join(output_dir, output_file_name), "recreate")
    tfile.cd()
    he3_spectrum.Write()
    he4_spectrum.Write()
    th1_pt_h3l.Write()
    th1_pt_h4l.Write()
    th2_n_sigma_he3_pt_mc.Write()
    th2_n_sigma_he4_pt_mc.Write()
    th2_n_sigma_he3_pt_data.Write()
    th2_n_sigma_he4_pt_data.Write()
    th2_wrong_mass_h3l_pt_mc.Write()
    th2_wrong_mass_h4l_pt_mc.Write()
    th2_dedx_tpcmom.Write()
    th2_dedx_tpcmom_mc_h3l.Write()
    th2_dedx_tpcmom_mc_h4l.Write()
    frame_h3l.Write()
    frame_h4l.Write()
    frame_h3l_wrong_mass.Write()
    frame_h4l_wrong_mass.Write()
    frame_data_h3l.Write()
    frame_data_h4l.Write()
    tfile.Close()


if __name__ == "__main__":
    main()

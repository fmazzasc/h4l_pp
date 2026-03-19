import argparse
import os
import re
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
ROOT.TH1.AddDirectory(False)

kOrangeC = ROOT.TColor.GetColor("#ff7f00")

# Keep all RooFit objects alive to prevent double-free at cleanup
_keep_alive = []

def main():
    parser = argparse.ArgumentParser(description="Configure the parameters of the script.")
    parser.add_argument("--config-file", dest="config_file", default="", help="Path to the YAML file with configuration.")
    parser.add_argument(
        "--chebychev-order",
        dest="chebychev_order",
        type=int,
        choices=(1, 2),
        default=None,
        help="Override the Chebyshev background order (1 or 2).",
    )
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
    chebychev_order = args.chebychev_order if args.chebychev_order is not None else int(config.get("chebychev_order", 2))
    if chebychev_order not in (1, 2):
        raise ValueError(f"Unsupported Chebyshev order: {chebychev_order}. Expected 1 or 2.")

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

    mc_rdf_h3l_full = mc_rdf_h3l_full.Filter('rej == 1')
    mc_rdf_h4l_full = mc_rdf_h4l_full.Filter('rej == 1')
    mc_rdf_h3l_base_reco = mc_rdf_h3l_full.Filter("fIsReco == true")
    mc_rdf_h4l_base_reco = mc_rdf_h4l_full.Filter("fIsReco == true")
    data_rdf = data_rdf.Filter(selections_string)
    mc_rdf_h3l = mc_rdf_h3l_base_reco.Filter(selections_string)
    mc_rdf_h4l = mc_rdf_h4l_base_reco.Filter(selections_string)

    print("-------------------------")
    print("Data filtered with selections. Generating QA histograms...")

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
    mc_rdf_h3l_no_pid = mc_rdf_h3l_base_reco.Filter(selections_string_no_pid).Filter(mass_window_sel)
    mc_rdf_h4l_no_pid = mc_rdf_h4l_base_reco.Filter(selections_string_no_pid).Filter(mass_window_sel)

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

    signal_h3l, pars_h3l, frame_h3l = utils.build_and_fit_dscb(
        "h3l", mass3HL, mass_roo_mc_h3l, (mass3_min, mass3_max), (0.001, 0.0024))
    signal_h4l, pars_h4l, frame_h4l = utils.build_and_fit_dscb(
        "h4l", mass4HL, mass_roo_mc_h4l, (3.91, 3.95), (0.001, 0.004))
    mu3HL, sigma_h3l = pars_h3l[0], pars_h3l[1]
    mu4HL, sigma_h4l = pars_h4l[0], pars_h4l[1]

    pdf_h3l_wrong_mass, mass_roo_mc_h3l_wrong_mass, frame_h3l_wrong_mass = utils.build_wrong_mass_pdf(
        "h3l", mass3HL, mc_rdf_h4l_no_pid, "fMassH3L")
    pdf_h4l_wrong_mass, mass_roo_mc_h4l_wrong_mass, frame_h4l_wrong_mass = utils.build_wrong_mass_pdf(
        "h4l", mass4HL, mc_rdf_h3l_no_pid, "fMassH4L")

    mass3HL.setBins(30)
    mass4HL.setBins(20)

    bkg_h3l = utils.build_chebychev("h3l", mass3HL, order=chebychev_order)
    bkg_h4l = utils.build_chebychev("h4l", mass4HL, order=chebychev_order)

    ## Simultaneous fit
    nsig_h3l = ROOT.RooRealVar("nsig_h3l", "signal events", 100, 0, 1e5)
    n_sig_h4l = ROOT.RooRealVar("nsig_h4l", "signal events", 100, 0, 1e5)
    n_bkg = ROOT.RooRealVar("nbkg", "background events", 100, 0, 1e5)

    model_h3l = ROOT.RooAddPdf("model_h3l", "model_h3l",
        ROOT.RooArgList(signal_h3l, pdf_h3l_wrong_mass, bkg_h3l),
        ROOT.RooArgList(nsig_h3l, n_sig_h4l, n_bkg))
    model_h4l = ROOT.RooAddPdf("model_h4l", "model_h4l",
        ROOT.RooArgList(signal_h4l, pdf_h4l_wrong_mass, bkg_h4l),
        ROOT.RooArgList(n_sig_h4l, nsig_h3l, n_bkg))

    mass_roo_data_h3l = utils.rdf_to_roodataset(data_rdf, "fMassH3L", mass3HL, "histo_data_h3l")
    mass_roo_data_h4l = utils.rdf_to_roodataset(data_rdf, "fMassH4L", mass4HL, "histo_data_h4l")

    categories = ROOT.RooCategory("categories", "categories")
    categories.defineType("h3l")
    categories.defineType("h4l")
    data = ROOT.RooDataSet("data", "data",
        ROOT.RooArgSet(mass3HL, mass4HL, categories),
        ROOT.RooFit.Index(categories),
        ROOT.RooFit.Import("h3l", mass_roo_data_h3l),
        ROOT.RooFit.Import("h4l", mass_roo_data_h4l))
    roosim = ROOT.RooSimultaneous("roosim", "roosim", categories)
    roosim.addPdf(model_h3l, "h3l")
    roosim.addPdf(model_h4l, "h4l")
    roosim.fitTo(data, ROOT.RooFit.Extended(True), ROOT.RooFit.Save(True))

    ## Signal extraction
    sig_h3l_val, sig_h3l_err = utils.integrate_in_signal_range(signal_h3l, mass3HL, mu3HL, sigma_h3l, nsig_h3l)
    sig_h4l_val, sig_h4l_err = utils.integrate_in_signal_range(signal_h4l, mass4HL, mu4HL, sigma_h4l, n_sig_h4l)
    bkg_h3l_val, bkg_h3l_err = utils.integrate_pdf(bkg_h3l, mass3HL, n_bkg)
    bkg_h4l_val, bkg_h4l_err = utils.integrate_pdf(bkg_h4l, mass4HL, n_bkg)
    wm_h3l_val, wm_h3l_err = utils.integrate_pdf(pdf_h3l_wrong_mass, mass3HL, n_sig_h4l)
    wm_h4l_val, wm_h4l_err = utils.integrate_pdf(pdf_h4l_wrong_mass, mass4HL, nsig_h3l)

    s_b_ratio_h3l, s_b_ratio_h3l_err = utils.s_over_b(sig_h3l_val, bkg_h3l_val, wm_h3l_val, sig_h3l_err, bkg_h3l_err, wm_h3l_err)
    s_b_ratio_h4l, s_b_ratio_h4l_err = utils.s_over_b(sig_h4l_val, bkg_h4l_val, wm_h4l_val, sig_h4l_err, bkg_h4l_err, wm_h4l_err)

    pinfo_h3l = utils.make_fit_pavetext(sig_h3l_val, sig_h3l_err, s_b_ratio_h3l, s_b_ratio_h3l_err, mu3HL, sigma_h3l)
    pinfo_h4l = utils.make_fit_pavetext(sig_h4l_val, sig_h4l_err, s_b_ratio_h4l, s_b_ratio_h4l_err, mu4HL, sigma_h4l)

    ## Efficiency
    print("-------------------------")
    print("Signal extraction completed. Starting yield computation")
    br_h3l = 0.25
    br_h4l = 0.55

    pt_sel = all_selections["fPt"]
    pt_range = [float(i) for i in pt_sel.split() if i.replace('.', '', 1).isdigit()]
    gen_pt_filter = f"fAbsGenPt >= {pt_range[0]} && fAbsGenPt < {pt_range[1]}"
    if is_matter == "matter":
        gen_pt_filter += " && fIsMatter == true"
    elif is_matter == "antimatter":
        gen_pt_filter += " && fIsMatter == false"

    mc_rdf_h3l_gen = mc_rdf_h3l_full.Filter(gen_pt_filter)
    mc_rdf_h4l_gen = mc_rdf_h4l_full.Filter(gen_pt_filter)

    # Book all counts lazily, trigger single event loop
    count_h3l_num = mc_rdf_h3l.Count()
    count_h4l_num = mc_rdf_h4l.Count()
    count_h3l_den = mc_rdf_h3l_gen.Count()
    count_h4l_den = mc_rdf_h4l_gen.Count()

    eff_3hl_mc = count_h3l_num.GetValue() / count_h3l_den.GetValue()
    eff_4hl_mc = count_h4l_num.GetValue() / count_h4l_den.GetValue()

    h_eff_h3l_pt = mc_rdf_h3l.Histo1D(("h_eff_h3l_pt", "h_eff_h3l_pt", 10, pt_range[0], pt_range[1]), "fPt")
    h_eff_h4l_pt = mc_rdf_h4l.Histo1D(("h_eff_h4l_pt", "h_eff_h4l_pt", 10, pt_range[0], pt_range[1]), "fPt")
    h_pt_gen_h3l = mc_rdf_h3l_gen.Histo1D(("h_pt_gen_h3l", "h_pt_gen_h3l", 10, pt_range[0], pt_range[1]), "fAbsGenPt")
    h_pt_gen_h4l = mc_rdf_h4l_gen.Histo1D(("h_pt_gen_h4l", "h_pt_gen_h4l", 10, pt_range[0], pt_range[1]), "fAbsGenPt")
    h_eff_h3l_pt.Divide(h_pt_gen_h3l.GetPtr())
    h_eff_h4l_pt.Divide(h_pt_gen_h4l.GetPtr())

    ## Plot data fit results
    frame_data_h3l = utils.plot_data_fit(mass3HL, mass_roo_data_h3l, model_h3l, "cb_h3l", "mc_pdf_h3l_wrong_mass", "bkg_h3l", pinfo_h3l, "frame_data_h3l")
    frame_data_h4l = utils.plot_data_fit(mass4HL, mass_roo_data_h4l, model_h4l, "cb_h4l", "mc_pdf_h4l_wrong_mass", "bkg_h4l", pinfo_h4l, "frame_data_h4l")

    print("-------------------------")
    print(f"efficiency h3l: {eff_3hl_mc}")
    print(f"efficiency h4l: {eff_4hl_mc}")
    print("nsig_h3l: ", nsig_h3l.getVal(), nsig_h3l.getError())
    print("nsig_h4l: ", n_sig_h4l.getVal(), n_sig_h4l.getError())
    print("h3l / h4l: ", nsig_h3l.getVal() / n_sig_h4l.getVal() / (eff_3hl_mc / eff_4hl_mc) / (br_h3l / br_h4l))

    tfile = ROOT.TFile.Open(os.path.join(output_dir, output_file_name), "recreate")
    tfile.cd()
    tfile.mkdir("qa")
    tfile.cd("qa")
    he3_spectrum.Write()
    he4_spectrum.Write()
    h_eff_h3l_pt.Write()
    h_eff_h4l_pt.Write()
    h_pt_gen_h3l.Write()
    h_pt_gen_h4l.Write()
    th1_pt_h3l.Write()
    th1_pt_h4l.Write()
    th2_dedx_tpcmom.Write()
    th2_dedx_tpcmom_mc_h3l.Write()
    th2_dedx_tpcmom_mc_h4l.Write()

    tfile.mkdir("templates")
    tfile.cd("templates")
    th2_n_sigma_he3_pt_mc.Write()
    th2_n_sigma_he4_pt_mc.Write()
    th2_n_sigma_he3_pt_data.Write()
    th2_n_sigma_he4_pt_data.Write()
    th2_wrong_mass_h3l_pt_mc.Write()
    th2_wrong_mass_h4l_pt_mc.Write()

    tfile.cd()
    frame_h3l.Write()
    frame_h4l.Write()
    frame_h3l_wrong_mass.Write()
    frame_h4l_wrong_mass.Write()
    frame_data_h3l.Write()
    frame_data_h4l.Write()
    tfile.Close()


if __name__ == "__main__":
    main()
    # Clear keep-alive after main returns but before ROOT cleanup
    _keep_alive.clear()
    ROOT.gROOT.GetListOfFiles().Clear()
    ROOT.gROOT.GetListOfCanvases().Clear()

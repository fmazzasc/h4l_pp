import re
import uuid
import atexit
import numpy as np
import ROOT

np.random.seed(1995)

ROOT.gInterpreter.Declare(
    """
#include <cmath>
#include <stdexcept>
#include <string>
#include <TRandom3.h>
#include <TROOT.h>
#include <TF1.h>
#include <vector>

double heBB(double rigidity, double mass, bool isMC) {
    double p1;
    double p2;
    double p3;
    double p4;
    double p5;

    if (isMC) {
        p1 = -831.8102679813016;
        p2 = 1.6703373067602871;
        p3 = 2.710231559185544;
        p4 = 0.5010996589484132;
        p5 = 2.160825981543478;
    } else {
        p1 = -321.34;
        p2 = 0.6539;
        p3 = 1.591;
        p4 = 0.8225;
        p5 = 2.363;
    }

    const double betagamma = rigidity * 2. / mass;
    const double beta = betagamma / std::sqrt(1. + betagamma * betagamma);
    const double aa = std::pow(beta, p4);
    const double bb = std::log(p3 + std::pow(1. / betagamma, p5));
    return (p2 - aa - bb) * p1 / aa;
}

float computeNSigmaHe4(float tpcMomHe, float tpcSignalHe, bool isMC) {
    const auto expBB = heBB(tpcMomHe, 3.727, isMC);
    return (tpcSignalHe - expBB) / (0.09f * tpcSignalHe);
}

float computeNSigmaHe3(float tpcMomHe, float tpcSignalHe, bool isMC) {
    const auto expBB = heBB(tpcMomHe, 2.80839160743, isMC);
    return (tpcSignalHe - expBB) / (0.09f * tpcSignalHe);
}

float avgClusterSize(unsigned int clusterSizes) {
    float clSizeAvg = 0.f;
    float nHits = 0.f;
    for (int iLayer = 0; iLayer < 7; ++iLayer) {
        const auto clSize = (clusterSizes >> (4 * iLayer)) & 0b1111;
        clSizeAvg += clSize;
        nHits += clSize > 0 ? 1.f : 0.f;
    }
    return clSizeAvg / (nHits + 1e-10f);
}

float nITSHits(unsigned int clusterSizes) {
    float nHits = 0.f;
    for (int iLayer = 0; iLayer < 7; ++iLayer) {
        nHits += (((clusterSizes >> (4 * iLayer)) & 0b1111) > 0) ? 1.f : 0.f;
    }
    return nHits;
}

int rejectionFlag(double value, const char* funcName, double maxValue) {
    auto* obj = gROOT->GetFunction(funcName);
    if (obj == nullptr) {
        throw std::runtime_error(std::string("Function not found: ") + funcName);
    }
    auto* func = dynamic_cast<TF1*>(obj);
    if (func == nullptr) {
        throw std::runtime_error(std::string("Object is not a TF1: ") + funcName);
    }
    static thread_local TRandom3 rng(0);
    const double frac = func->Eval(value) / maxValue;
    return rng.Rndm() > frac ? -1 : 1;
}
"""
)


# ---------------------------------------------------------------------------
# Global list to prevent RooFit objects from being garbage-collected
# ---------------------------------------------------------------------------
_keep_alive = []


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def convert_sel_to_string(selection):
    sel_string = ""
    conj = " and "
    for _, val in selection.items():
        sel_string = sel_string + val + conj
    return sel_string[:-len(conj)]


def convert_sel_to_rdf_string(selection):
    sel_string = convert_sel_to_string(selection)
    sel_string = re.sub(r"\band\b", "&&", sel_string)
    sel_string = re.sub(r"\bor\b", "||", sel_string)
    sel_string = re.sub(r"\bTrue\b", "true", sel_string)
    sel_string = re.sub(r"\bFalse\b", "false", sel_string)
    return sel_string


# ---------------------------------------------------------------------------
# RDataFrame / ROOT helpers
# ---------------------------------------------------------------------------
def build_chain(file_names, tree_name, folder_prefix="DF_"):
    chain = ROOT.TChain(tree_name)
    for file_name in file_names:
        root_file = ROOT.TFile.Open(file_name)
        if not root_file or root_file.IsZombie():
            print(f"Could not open file: {file_name}")
            continue
        for key in root_file.GetListOfKeys():
            key_name = key.GetName()
            if folder_prefix in key_name:
                chain.Add(f"{file_name}/{key_name}/{tree_name}")
        root_file.Close()
    return chain


def redefine_or_define(rdf, column_name, expression):
    column_names = {str(col) for col in rdf.GetColumnNames()}
    if column_name in column_names:
        return rdf.Redefine(column_name, expression)
    return rdf.Define(column_name, expression)


def clone_result_hist(result_ptr, name):
    histo = result_ptr.GetValue().Clone(name)
    histo.SetDirectory(0)
    return histo


def rdf_to_array(rdf, column):
    return np.asarray(rdf.AsNumpy([column])[column], dtype=np.float64)


def rdf_to_roodataset(rdf, column, roo_var, name="data"):
    """Convert an RDF column directly to a RooDataSet without a Python loop."""
    arr = rdf.AsNumpy([column])[column]
    return ROOT.RooDataSet.from_numpy(
        {roo_var.GetName(): arr}, ROOT.RooArgSet(roo_var), name=name
    )


# ---------------------------------------------------------------------------
# RooFit model builders
# ---------------------------------------------------------------------------
def build_and_fit_dscb(name, mass_var, mass_dataset, mu_range, sigma_range):
    """Build a double-sided Crystal Ball, fit to MC, freeze shape params."""
    mu = ROOT.RooRealVar(f"mu_{name}", "hypernucl mass", *mu_range, "GeV/c^{2}")
    sigma = ROOT.RooRealVar(f"sigma_{name}", "hypernucl width", *sigma_range, "GeV/c^{2}")
    a1 = ROOT.RooRealVar(f"a1_{name}", f"a1_{name}", 0., 5.)
    a2 = ROOT.RooRealVar(f"a2_{name}", f"a2_{name}", 0., 5.)
    n1 = ROOT.RooRealVar(f"n1_{name}", f"n1_{name}", 0., 5.)
    n2 = ROOT.RooRealVar(f"n2_{name}", f"n2_{name}", 0., 5.)
    pars = [mu, sigma, a1, a2, n1, n2]
    cb = ROOT.RooCrystalBall(f"cb_{name}", f"cb_{name}", mass_var, mu, sigma, a1, n1, a2, n2)
    cb.fitTo(mass_dataset)
    frame = mass_var.frame()
    frame.SetName(f"frame_{name}_mc")
    mass_dataset.plotOn(frame)
    cb.plotOn(frame)
    # Fix shape params; allow mu to float; allow sigma to widen up to 20%
    for par in pars:
        if "mu_" in par.GetName():
            continue
        if "sigma_" in par.GetName():
            par.setRange(par.getVal(), par.getVal() * 1.2)
            continue
        par.setConstant(True)
    _keep_alive.extend(pars + [cb])
    return cb, pars, frame


def build_wrong_mass_pdf(name, mass_var, mc_rdf, mass_col, smooth_width=0.003):
    """Build a smoothed wrong-mass template via FFT convolution of a histogram PDF."""
    mass_var.setBins(30)
    ds = rdf_to_roodataset(mc_rdf, mass_col, mass_var, f"histo_mc_{name}_wrong_mass")
    dh = ROOT.RooDataHist(f"dh_{name}_wrong_mass", f"dh_{name}_wrong_mass",
                          ROOT.RooArgList(mass_var), ds)
    histpdf = ROOT.RooHistPdf(f"histpdf_{name}_wrong_mass", f"histpdf_{name}_wrong_mass",
                              ROOT.RooArgSet(mass_var), dh, 0)
    smooth_mean = ROOT.RooRealVar(f"smooth_mean_{name}", "smooth_mean", 0.0)
    smooth_sigma = ROOT.RooRealVar(f"smooth_width_{name}", "smooth_width", smooth_width)
    gauss = ROOT.RooGaussian(f"gauss_kernel_{name}", f"gauss_kernel_{name}",
                             mass_var, smooth_mean, smooth_sigma)
    mass_var.setBins(10000, "cache")
    pdf = ROOT.RooFFTConvPdf(f"mc_pdf_{name}_wrong_mass", f"mc_pdf_{name}_wrong_mass",
                             mass_var, histpdf, gauss)
    pdf.setBufferFraction(0.2)
    frame = mass_var.frame()
    frame.SetName(f"frame_{name}_wrong_mass")
    ds.plotOn(frame)
    pdf.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed))
    _keep_alive.extend([ds, dh, histpdf, smooth_mean, smooth_sigma, gauss, pdf])
    return pdf, ds, frame


def build_chebychev(name, mass_var, order=2):
    """Build a 1st- or 2nd-order Chebyshev background PDF."""
    if order not in (1, 2):
        raise ValueError(f"Unsupported Chebyshev order: {order}. Expected 1 or 2.")

    coeffs = ROOT.RooArgList()
    keep_alive = []
    for idx in range(order):
        coeff = ROOT.RooRealVar(f"c{idx}_bkg_{name}", f"c{idx}_bkg_{name}", -1, 1.0)
        coeffs.add(coeff)
        keep_alive.append(coeff)

    cheb = ROOT.RooChebychev(f"bkg_{name}", f"bkg_{name}", mass_var, coeffs)
    _keep_alive.extend([*keep_alive, coeffs, cheb])
    return cheb


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------
def integrate_pdf(pdf, mass_var, norm_var, range_name=None):
    """Integrate a PDF (optionally in a named range) and scale by norm_var."""
    if range_name:
        integral = pdf.createIntegral(ROOT.RooArgSet(mass_var), ROOT.RooArgSet(mass_var), range_name)
    else:
        integral = pdf.createIntegral(ROOT.RooArgSet(mass_var), ROOT.RooArgSet(mass_var))
    val = integral.getVal() * norm_var.getVal()
    err = val * norm_var.getError() / norm_var.getVal() if norm_var.getVal() != 0 else 0
    return val, err


def integrate_in_signal_range(pdf, mass_var, mu, sigma, norm_var, n_sigma=3):
    """Integrate a PDF in a ±n_sigma window around the mean."""
    mass_var.setRange("signal", mu.getVal() - n_sigma * sigma.getVal(),
                      mu.getVal() + n_sigma * sigma.getVal())
    return integrate_pdf(pdf, mass_var, norm_var, "signal")


def s_over_b(sig, bkg, wm, sig_err, bkg_err, wm_err):
    """Compute signal-over-background ratio with error propagation."""
    ratio = sig / (bkg + wm)
    ratio_err = ratio * np.sqrt((sig_err / sig) ** 2 + (bkg_err / bkg) ** 2 + (wm_err / wm) ** 2)
    return ratio, ratio_err


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def make_fit_pavetext(sig_val, sig_err, sb_ratio, sb_err, mu, sigma):
    """Create a TPaveText with fit results."""
    kOrangeC = ROOT.TColor.GetColor('#ff7f00')
    pinfo = ROOT.TPaveText(0.632, 0.5, 0.932, 0.85, "NDC")
    pinfo.SetBorderSize(0)
    pinfo.SetFillStyle(0)
    pinfo.SetTextAlign(11)
    pinfo.SetTextFont(42)
    pinfo.AddText(f"Signal (S): {sig_val:.0f} #pm {sig_err:.0f}")
    pinfo.AddText(f"S/B (3 #sigma): {sb_ratio:.1f} #pm {sb_err:.1f}")
    pinfo.AddText("#mu = " + f"{mu.getVal() * 1e3:.2f} #pm {mu.getError() * 1e3:.2f}" + " MeV/#it{c}^{2}")
    pinfo.AddText("#sigma = " + f"{sigma.getVal() * 1e3:.2f} #pm {sigma.getError() * 1e3:.2f}" + " MeV/#it{c}^{2}")
    return pinfo


def plot_data_fit(mass_var, dataset, model, signal_name, wrong_mass_name, bkg_name, pinfo, frame_name):
    """Plot data with overlaid fit components."""
    kOrangeC = ROOT.TColor.GetColor('#ff7f00')
    frame = mass_var.frame()
    frame.SetName(frame_name)
    dataset.plotOn(frame)
    model.plotOn(frame)
    model.plotOn(frame, ROOT.RooFit.Components(signal_name), ROOT.RooFit.LineColor(kOrangeC), ROOT.RooFit.LineStyle(2))
    model.plotOn(frame, ROOT.RooFit.Components(wrong_mass_name), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.LineStyle(2))
    model.plotOn(frame, ROOT.RooFit.Components(bkg_name), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(2))
    frame.addObject(pinfo)
    return frame


# ---------------------------------------------------------------------------
# RDF column definitions / corrections
# ---------------------------------------------------------------------------
def correct_and_convert_rdf(rdf, calibrate_he3_pt=False, isMC=False, isH4L=False, pt_spectrum=None):
    kDefaultPID = 15
    kPionPID = 2
    kTritonPID = 6
    column_names = {str(col) for col in rdf.GetColumnNames()}

    if "fFlags" in column_names:
        rdf = rdf.Define("fHePIDHypo", "(fFlags >> 4)")
        rdf = rdf.Define("fPiPIDHypo", "(fFlags & 0b1111)")
    else:
        rdf = rdf.Define("fHePIDHypo", str(kDefaultPID))
        rdf = rdf.Define("fPiPIDHypo", str(kPionPID))

    if "fTPCChi2He" not in column_names or isMC:
        rdf = redefine_or_define(rdf, "fTPCChi2He", "1.f")
    else:
        rdf = redefine_or_define(rdf, "fTPCChi2He", "std::isnan(fTPCChi2He) ? 1.f : fTPCChi2He")

    if calibrate_he3_pt:
        no_pid_count = rdf.Filter(
            f"fHePIDHypo != {kDefaultPID} && fHePIDHypo != {kPionPID}"
        ).Count().GetValue()
        if no_pid_count == 0:
            print("PID in tracking not detected, using old momentum re-calibration")
            rdf = rdf.Redefine(
                "fPtHe3",
                "fPtHe3 + 2.98019e-02 + 7.66100e-01 * exp(-1.31641e+00 * fPtHe3)",
            )
        else:
            print("PID in tracking detected, using new momentum re-calibration")
            rdf = rdf.Redefine(
                "fPtHe3",
                f"fHePIDHypo == {kTritonPID} ? "
                "(fPtHe3 - 0.1286 - 0.1269 * fPtHe3 + 0.06 * fPtHe3 * fPtHe3) : fPtHe3",
            )

    rdf = rdf.Define("fPxHe3", "fPtHe3 * cos(fPhiHe3)")
    rdf = rdf.Define("fPyHe3", "fPtHe3 * sin(fPhiHe3)")
    rdf = rdf.Define("fPzHe3", "fPtHe3 * sinh(fEtaHe3)")
    rdf = rdf.Define("fPHe3", "fPtHe3 * cosh(fEtaHe3)")
    rdf = rdf.Define("fEnHe3", "sqrt(fPHe3 * fPHe3 + 2.8083916 * 2.8083916)")
    rdf = rdf.Define("fEnHe4", "sqrt(fPHe3 * fPHe3 + 3.7273794 * 3.7273794)")
    rdf = rdf.Define("fPxPi", "fPtPi * cos(fPhiPi)")
    rdf = rdf.Define("fPyPi", "fPtPi * sin(fPhiPi)")
    rdf = rdf.Define("fPzPi", "fPtPi * sinh(fEtaPi)")
    rdf = rdf.Define("fPPi", "fPtPi * cosh(fEtaPi)")
    rdf = rdf.Define("fEnPi", "sqrt(fPPi * fPPi + 0.139570 * 0.139570)")
    rdf = rdf.Define("fPx", "fPxHe3 + fPxPi")
    rdf = rdf.Define("fPy", "fPyHe3 + fPyPi")
    rdf = rdf.Define("fPz", "fPzHe3 + fPzPi")
    rdf = rdf.Define("fP", "sqrt(fPx * fPx + fPy * fPy + fPz * fPz)")
    rdf = rdf.Define("fEn", "fEnHe3 + fEnPi")
    rdf = rdf.Define("fEn4", "fEnHe4 + fEnPi")
    rdf = rdf.Define("fPt", "sqrt(fPx * fPx + fPy * fPy)")
    rdf = rdf.Define("fEta", "std::acosh(fP / fPt)")
    rdf = rdf.Define("fCosLambda", "fPt / fP")
    rdf = rdf.Define("fCosLambdaHe", "fPtHe3 / fPHe3")
    rdf = rdf.Define("fNSigmaHe4", f"computeNSigmaHe4(fTPCmomHe, fTPCsignalHe, {str(isMC).lower()})")
    rdf = rdf.Define("fNSigmaHe3", f"computeNSigmaHe3(fTPCmomHe, fTPCsignalHe, {str(isMC).lower()})")
    rdf = rdf.Define("fDecLen", "sqrt(fXDecVtx * fXDecVtx + fYDecVtx * fYDecVtx + fZDecVtx * fZDecVtx)")
    rdf = rdf.Define("fCt", "fDecLen * 3.922 / fP" if isH4L else "fDecLen * 2.99131 / fP")
    rdf = rdf.Define("fDecRad", "sqrt(fXDecVtx * fXDecVtx + fYDecVtx * fYDecVtx)")
    rdf = rdf.Define("fCosPA", "(fPx * fXDecVtx + fPy * fYDecVtx + fPz * fZDecVtx) / (fP * fDecLen)")
    rdf = rdf.Define("fMassH3L", "sqrt(fEn * fEn - fP * fP)")
    rdf = rdf.Define("fMassH4L", "sqrt(fEn4 * fEn4 - fP * fP)")
    rdf = rdf.Define("fTPCSignMomHe3", "fTPCmomHe * (fIsMatter ? 1.f : -1.f)")
    rdf = rdf.Define("fGloSignMomHe3", "fPHe3 / 2.f * (fIsMatter ? 1.f : -1.f)")

    if "fITSclusterSizesHe" in column_names:
        rdf = rdf.Define("fAvgClusterSizeHe", "avgClusterSize(fITSclusterSizesHe)")
        rdf = rdf.Define("fAvgClusterSizePi", "avgClusterSize(fITSclusterSizesPi)")
        rdf = rdf.Define("nITSHitsHe", "nITSHits(fITSclusterSizesHe)")
        rdf = rdf.Define("nITSHitsPi", "nITSHits(fITSclusterSizesPi)")
        rdf = rdf.Define("fAvgClSizeCosLambda", "fAvgClusterSizeHe * fCosLambdaHe")

    if "fPsiFT0C" in column_names:
        rdf = rdf.Define("fPhi", "atan2(fPy, fPx)")
        rdf = rdf.Define("fV2", "cos(2.f * (fPhi - fPsiFT0C))")

    if isMC:
        rdf = rdf.Define("fGenDecLen", "sqrt(fGenXDecVtx * fGenXDecVtx + fGenYDecVtx * fGenYDecVtx + fGenZDecVtx * fGenZDecVtx)")
        rdf = rdf.Define("fGenDecRad", "sqrt(fGenXDecVtx * fGenXDecVtx + fGenYDecVtx * fGenYDecVtx)")
        rdf = rdf.Define("fGenPz", "fGenPt * sinh(fGenEta)")
        rdf = rdf.Define("fGenP", "sqrt(fGenPt * fGenPt + fGenPz * fGenPz)")
        rdf = rdf.Define("fAbsGenPt", "std::abs(fGenPt)")
        rdf = rdf.Define("fGenCt", "fGenDecLen * 3.922 / fGenP" if isH4L else "fGenDecLen * 2.99131 / fGenP")
        if "fIsTwoBodyDecay" in column_names:
            rdf = rdf.Filter("fIsTwoBodyDecay == true")
        if pt_spectrum is not None:
            func_name, max_bw = _register_rejection_distribution(pt_spectrum, "rej")
            rdf = rdf.Define("rej", f'rejectionFlag(fAbsGenPt, "{func_name}", {max_bw})')

    return rdf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _register_rejection_distribution(distribution, flag_name):
    """Register a ROOT function globally and return (func_name, max_value)."""
    func_name = f"g_{flag_name}_{uuid.uuid4().hex}"
    distribution = distribution.Clone(func_name)
    distribution.SetName(func_name)
    if not hasattr(ROOT, "_rejection_functions"):
        ROOT._rejection_functions = {}
    ROOT._rejection_functions[func_name] = distribution
    ROOT.gROOT.GetListOfFunctions().Add(distribution)
    max_bw = distribution.GetMaximum()
    return func_name, max_bw


def _cleanup_rejection_functions():
    """Remove registered TF1s from ROOT's global list to prevent double-delete at exit."""
    if hasattr(ROOT, "_rejection_functions"):
        func_list = ROOT.gROOT.GetListOfFunctions()
        for func in ROOT._rejection_functions.values():
            func_list.Remove(func)
        ROOT._rejection_functions.clear()


atexit.register(_cleanup_rejection_functions)

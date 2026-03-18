# Hyperhydrogen-4 analysis in pp collisions

Combined simultaneous invariant mass fit of ${}^{3}_{\\Lambda}\text{H}$ and ${}^{4}_{\\Lambda}\text{H}$ in pp collisions at $\sqrt{s} = 13.6$ TeV with ALICE Run 3 data, using ROOT RDataFrame.

## Project structure

```
h4l_pp/
├── configs/                    # YAML configuration files
│   └── config_pp_h4l_rdf.yaml
├── utils/
│   └── utils.py                # Utility functions (RDF helpers, RooDataSet conversion, selections)
├── fit_h3l_h4l_rdf.py          # Main script: simultaneous H3L + H4L invariant mass fit
├── tpc_calibration_rdf.py      # TPC dE/dx calibration via Bethe-Bloch parametrisation
└── README.md
```

## Analysis workflow

1. **TPC calibration** — Fit the Bethe-Bloch parametrisation to the TPC dE/dx vs rigidity distribution for ${}^{3}\text{He}$ and ${}^{4}\text{He}$ in data and MC.
2. **Signal extraction** — Simultaneous unbinned maximum-likelihood fit of the H3L and H4L invariant mass spectra, accounting for:
   - Signal: double-sided Crystal Ball (shape from MC, $\mu$ and $\sigma$ floating in data)
   - Wrong-mass contamination: FFT-smoothed histogram templates from MC (no PID selection for maximum statistics)
   - Combinatorial background: exponential PDF

## Usage

```bash
# TPC dE/dx calibration
python tpc_calibration_rdf.py --config-file configs/config_tpc_calib.yaml

# Invariant mass fit
python fit_h3l_h4l_rdf.py --config-file configs/config_pp_h4l_rdf.yaml
```

## Configuration

All analysis parameters are set in YAML config files:

| Key | Description |
|---|---|
| `input_files_data` | Paths to data AOD files |
| `input_files_mc_h3l` | Paths to H3L MC AOD files |
| `input_files_mc_h4l` | Paths to H4L MC AOD files |
| `output_dir` | Output directory for results |
| `output_file` | Output ROOT file name |
| `selection` | Topological and kinematic cuts |
| `pid_selection` | PID cuts (applied to data and signal MC, **not** to wrong-mass templates) |
| `colliding_system` | Collision system (`pp`, `PbPb`) |
| `is_matter` | Matter/antimatter selection (`matter`, `antimatter`, or empty for both) |
| `calibrate_he_momentum` | Whether to apply He momentum calibration |

## Dependencies

- ROOT ≥ 6.28 (with RDataFrame, RooFit, `RooDataSet.from_numpy`)
- Python ≥ 3.9
- NumPy
- PyYAML

## Notes

- Wrong-mass templates are built **without PID selections** to maximise MC statistics. The PID cuts are separated in the config under `pid_selection`.
- Template smoothing uses `RooFFTConvPdf` (histogram ⊗ Gaussian kernel) rather than `RooKeysPdf` for performance.
- For data, the TPC dE/dx slices are fitted with a double Gaussian (signal + background peaked at lower values), with the background mean constrained to be below the signal mean via reparametrisation ($\mu_\text{bkg} = \mu_\text{sig} - \Delta$, $\Delta \geq 0$).

# Observation of ${}^{4}_{\\Lambda}\text{H}$ in pp collisions at $\sqrt{s} = 13.6$ TeV with ALICE

Analysis code for the search for ${}^{4}_{\\Lambda}\text{H}$ (hyperhydrogen-4) in pp collisions at $\sqrt{s} = 13.6$ TeV using ALICE Run 3 data. The signal is extracted via a simultaneous unbinned fit of the ${}^{3}_{\\Lambda}\text{H}$ and ${}^{4}_{\\Lambda}\text{H}$ invariant mass spectra, where ${}^{3}_{\\Lambda}\text{H}$ serves as a reference signal.

## Project structure

```
h4l_pp/
├── configs/
│   ├── config_pp_h4l_rdf.yaml     # Invariant mass fit configuration
│   └── config_tpc_calib.yaml       # TPC dE/dx calibration configuration
├── utils/
│   └── utils.py                    # Utility functions (RDF helpers, RooDataSet tools, selections)
├── fit_h3l_h4l_rdf.py              # Simultaneous H3L + H4L invariant mass fit
├── tpc_calibration_rdf.py          # TPC dE/dx calibration (Bethe-Bloch fits)
└── README.md
```

## Analysis strategy

1. **TPC calibration** (`tpc_calibration_rdf.py`) — Bethe-Bloch parametrisation of the TPC dE/dx for ${}^{3}\text{He}$ and ${}^{4}\text{He}$ in data and MC. For data, a double-Gaussian fit (signal + low-dE/dx background) is used per momentum slice, with the background mean constrained below the signal mean.

2. **Signal extraction** (`fit_h3l_h4l_rdf.py`) — Simultaneous unbinned maximum-likelihood fit:
   - **Signal**: double-sided Crystal Ball (shape fixed from MC, $\mu$ and $\sigma$ floating in data)
   - **Wrong-mass contamination**: FFT-smoothed histogram templates from MC, built **without PID cuts** for maximum statistics
   - **Combinatorial background**: exponential PDF

## Usage

```bash
# Step 1: TPC dE/dx calibration
python tpc_calibration_rdf.py --config-file configs/config_tpc_calib.yaml

# Step 2: Invariant mass fit
python fit_h3l_h4l_rdf.py --config-file configs/config_pp_h4l_rdf.yaml
```

## Configuration

### `config_pp_h4l_rdf.yaml` — Fit configuration

| Key | Description |
|---|---|
| `input_files_data` | Paths to data AOD files |
| `input_files_mc_h3l` | Paths to ${}^{3}_{\\Lambda}\text{H}$ MC AOD files |
| `input_files_mc_h4l` | Paths to ${}^{4}_{\\Lambda}\text{H}$ MC AOD files |
| `output_dir` | Output directory for results |
| `output_file` | Output ROOT file name |
| `selection` | Topological and kinematic cuts |
| `pid_selection` | PID cuts (applied to data and signal MC only, **not** to wrong-mass templates) |
| `is_matter` | `matter`, `antimatter`, or empty for both |
| `calibrate_he_momentum` | Whether to apply ${}^{3}\text{He}$ momentum calibration |

### `config_tpc_calib.yaml` — Calibration configuration

| Key | Description |
|---|---|
| `input_files_data` | Paths to data AOD files |
| `input_files_mc` | Paths to MC AOD files |
| `p_bins` | Momentum binning for dE/dx slices |
| `dedx_bins` / `dedx_range` | dE/dx histogram binning |

## Dependencies

- ROOT ≥ 6.28 (RDataFrame, RooFit, `RooDataSet.from_numpy`)
- Python ≥ 3.9
- NumPy, PyYAML

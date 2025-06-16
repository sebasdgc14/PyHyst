# PyHyst: Python Magnetic Hysteresis Analysis Toolkit

**PyHyst** is an open-source Python toolkit and GUI for the analysis, correction, and modeling of magnetic hysteresis data. It provides a modern, extensible alternative to proprietary and MATLAB-based solutions, supporting advanced preprocessing, flexible fitting, and reproducible export of results.

---

## Features

- **User-friendly GUI** built with PyQt5 and Matplotlib
- **Flexible data loading**: Supports `.DAT` and `.CSV` files, with optional manual entry of sample mass/volume and unit conversion
- **Advanced preprocessing**:
  - Branch splitting (Diff/Zero methods)
  - Center shift correction (R² maximization)
  - Drift correction (Auto, Symmetric, Positive, Upper)
  - High-field slope correction (approach-to-saturation model)
- **Auxiliary curve computation**: Mih (induced) and Mrh (remanent) decomposition
- **Multiple fitting modes**:
  - Single (logistic/tanh)
  - Multi (adaptive basis, NNLS)
  - Diego (arctangent model)
  - Custom (user-defined functions via GUI)
- **Parameter extraction**: $M_s$, $M_r$, $H_c$ via robust interpolation
- **Export**: PDF report and corrected data CSV with all metadata, plots, and fit results

---

## Installation

1. **Clone the repository:**
   bash
   git clone https://github.com/sebasdgc14/PyHyst.git
   cd PyHyst
2. **Install dependencies:**
    pip install -r requirements.txt
3. **Run the GUI:**
    python PyHyst/PyHyst_GUI.py

## Usage
**Data Loading and Unit Conversion**
1. Launch the GUI and click the "Load .DAT or .CSV File" button to select your data file.
2. (Optional) Enter sample volume and/or mass in the provided fields. You can specify the original and target units for both volume (cm³, mm³, m³) and mass (g, mg, kg) using dropdown menus. The GUI will automatically convert your input to the desired units and display the converted value.
3. Set field and moment units using the respective dropdowns (Oe/A/m for field, emu/A·m² for moment). The software will handle all necessary conversions for consistent analysis.
4. Branch splitting method can be selected (Diff/Zero) to control how the upper and lower branches are defined.

**Corrections and Preprocessing**
1. Center Shift: Checkbox to apply centering and regular gridding.
2. Drift Correction: Dropdown to select drift correction mode (None, Auto, Symmetric, Positive, Upper).
3. Slope Correction: Checkbox to apply high-field slope correction.
4. Apply Corrections: Button to process all selected corrections.

**Fitting**
1. Fitting Mode: Dropdown to select Single, Multi, Diego, or Custom fit.
2. Custom/Diego Inputs: For Diego and Custom modes, enter parameter guesses and (for Custom) function definitions for each branch.
3. Fit Data: Button to run the selected fitting routine. Results are shown in Plot 3 and summarized in the parameter display panel.

**Visualization**
Plot 1: Original or corrected hysteresis branches.
Plot 2: Auxiliary curves (Mih and Mrh).
Plot 3: Fitting results, with data and fitted curves overlaid.

**Exporting Results**
1. Export Results: Button to save a PDF report (including plots, sample info, corrections, fit parameters, and metrics) and a CSV file with the corrected data.

## Example Workflow
1. Load data and set units.
2. Enter sample mass/volume if available.
3. Select branch splitting and apply corrections as needed.
4. Compute auxiliary curves.
5. Choose fitting mode and run fitting.
6. Inspect fit quality and extracted parameters.
7. Export results for reporting or further analysis.

## File Structure
PyHyst/
├── PyHyst_GUI.py         # Main GUI application
├── Read_Data.py          # Data loading and preprocessing
├── Auxiliary_Curves.py   # Mih/Mrh decomposition
├── Center_Shift.py       # Centering and gridding
├── Drift_Correction.py   # Drift correction routines
├── Slope_Correction.py   # High-field slope correction
├── Fitting.py            # Fitting and parameter extraction
├── Utilities.py          # Helper functions

## Exported Results
1. PDF Report: Includes sample info, corrections, fit plots, extracted parameters, and metrics.
2. CSV File: Contains corrected and processed data for further analysis.

## Citing PyHyst
Sebastian Diaz Granados Cano, et al. PyHyst: Python Magnetic Hysteresis Analysis Toolkit, (2025).
https://github.com/sebasdgc14/PyHyst

## License
GPL-3.0

## Acknowledgements
Methodology inspired by Jackson & Solheid (2010), Von Dobeneck (1996), and Paterson et al. (2018, HystLab).
Developed at Grupo de Estado Sólido, Universidad de Antioquia, Medellín, Colombia.

## Questions, Reports, Contributions
For questions, bug reports, or contributions, please open an issue or pull request on GitHub.


import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

import logging
from logging import getLogger
from Read_Data import Read_Data
from Auxiliary_Curves import Auxiliary_Curves
from Center_Shift import Center_Shift
from Drift_Correction import Drift_Correction
from Slope_Correction import Slope_Correction
from Fitting import Fitting

logger = getLogger(__name__)


class PyHystGUI(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyHyst GUI")
        self.setGeometry(100, 100, 1200, 900)
        # Set the main layout
        grid = qtw.QGridLayout()
        self.setLayout(grid)

        controls = qtw.QVBoxLayout()
        controls.setSpacing(0)
        controls.setContentsMargins(0, 0, 0, 0)

        corrections = qtw.QVBoxLayout()
        corrections.setSpacing(0)
        corrections.setContentsMargins(0, 0, 0, 0)

        fittings = qtw.QVBoxLayout()
        fittings.setSpacing(0)
        fittings.setContentsMargins(0, 0, 0, 0)

        # ------------------------------- Regarding file loading ------------------------------- #
        # File load button
        self.load_button = qtw.QPushButton("Load .DAT or .CSV File")
        self.load_button.clicked.connect(self.load_file)
        controls.addWidget(self.load_button)
        
        # Volume input + label + original unit + target unit + info
        vol_layout = qtw.QHBoxLayout()
        self.vol_input = qtw.QLineEdit()
        self.vol_input.setPlaceholderText("Enter Volume (optional)")
        self.vol_input.editingFinished.connect(self.reprocess_data)
        self.vol_unit_input = qtw.QComboBox()
        self.vol_unit_input.addItems(["cm**3", "mm**3", "m**3"])
        self.vol_unit_input.setFixedWidth(70)
        self.vol_target_unit_input = qtw.QComboBox()
        self.vol_target_unit_input.addItems(["cm**3", "mm**3", "m**3"])
        self.vol_target_unit_input.setFixedWidth(70)
        self.vol_target_label = qtw.QLabel(self.vol_target_unit_input.currentText())
        self.vol_label = qtw.QLabel("N/A")
        vol_info = qtw.QToolButton()
        vol_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        vol_info.setToolTip("Specify the original and target units for volume (e.g. 'cm**3' to 'mm**3').")
        self.vol_unit_input.currentIndexChanged.connect(self.reprocess_data)
        self.vol_target_unit_input.currentIndexChanged.connect(self.reprocess_data)

        # Update target label when dropdown changes
        self.vol_target_unit_input.currentIndexChanged.connect(
            lambda: self.vol_target_label.setText(self.vol_target_unit_input.currentText()))

        # Arrange: [input] [original unit] → [target unit] [=] [converted label] [info]
        vol_layout.addWidget(self.vol_input)
        vol_layout.addWidget(self.vol_unit_input)
        vol_layout.addWidget(qtw.QLabel("→"))
        vol_layout.addWidget(self.vol_target_unit_input)
        vol_layout.addWidget(qtw.QLabel("="))
        vol_layout.addWidget(self.vol_label)
        vol_layout.addWidget(self.vol_target_label)
        vol_layout.addWidget(vol_info)
        controls.addLayout(vol_layout)

        # Mass input + label + original unit + target unit + info
        mass_layout = qtw.QHBoxLayout()
        self.mass_input = qtw.QLineEdit()
        self.mass_input.setPlaceholderText("Enter Mass (optional)")
        self.mass_input.editingFinished.connect(self.reprocess_data)
        self.mass_unit_input = qtw.QComboBox()
        self.mass_unit_input.addItems(["g", "mg", "kg"])
        self.mass_unit_input.setFixedWidth(70)
        self.mass_target_unit_input = qtw.QComboBox()
        self.mass_target_unit_input.addItems(["g", "mg", "kg"])
        self.mass_target_unit_input.setFixedWidth(70)
        self.mass_label = qtw.QLabel("N/A")
        self.mass_target_label = qtw.QLabel(self.mass_target_unit_input.currentText())
        mass_info = qtw.QToolButton()
        mass_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        mass_info.setToolTip("Specify the original and target units for mass (e.g. 'g' to 'mg').")
        self.mass_unit_input.currentIndexChanged.connect(self.reprocess_data)
        self.mass_target_unit_input.currentIndexChanged.connect(self.reprocess_data)

        # Update target label when dropdown changes
        self.mass_target_unit_input.currentIndexChanged.connect(
            lambda: self.mass_target_label.setText(self.mass_target_unit_input.currentText())
        )

        # Arrange: [input] [original unit] → [target unit] [=] [converted label] [info]
        mass_layout.addWidget(self.mass_input)
        mass_layout.addWidget(self.mass_unit_input)
        mass_layout.addWidget(qtw.QLabel("→"))
        mass_layout.addWidget(self.mass_target_unit_input)
        mass_layout.addWidget(qtw.QLabel("="))
        mass_layout.addWidget(self.mass_label)
        mass_layout.addWidget(self.mass_target_label)
        mass_layout.addWidget(mass_info)
        controls.addLayout(mass_layout)

        # Field input + original unit + target unit + info
        field_layout = qtw.QHBoxLayout()
        self.field_unit_input = qtw.QComboBox()
        self.field_unit_input.addItems(["Oe", "A/m"])
        self.field_unit_input.setFixedWidth(70)
        self.field_target_unit_input = qtw.QComboBox()
        self.field_target_unit_input.addItems(["Oe", "A/m"])
        self.field_target_unit_input.setFixedWidth(70)
        self.field_label = qtw.QLabel("N/A")
        self.field_target_label = qtw.QLabel(self.field_target_unit_input.currentText())
        field_info = qtw.QToolButton()
        field_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        field_info.setToolTip("Specify the original and target units for the magnetic field (e.g. 'Oe' to 'A/m').")


        self.field_target_unit_input.currentIndexChanged.connect(
            lambda: self.field_target_label.setText(self.field_target_unit_input.currentText())
        )

        self.field_unit_input.currentIndexChanged.connect(self.reprocess_data)
        self.field_target_unit_input.currentIndexChanged.connect(self.reprocess_data)

        # Arrange: [original unit] → [target unit] [=] [converted label] [info]
        field_layout.addWidget(qtw.QLabel("Field:"))
        field_layout.addWidget(self.field_unit_input)
        field_layout.addWidget(qtw.QLabel("→"))
        field_layout.addWidget(self.field_target_unit_input)
        field_layout.addWidget(qtw.QLabel("="))
        field_layout.addWidget(self.field_label)
        field_layout.addWidget(self.field_target_label)
        field_layout.addWidget(field_info)
        controls.addLayout(field_layout)

        # Moment input + original unit + target unit + info
        moment_layout = qtw.QHBoxLayout()
        self.moment_unit_input = qtw.QComboBox()
        self.moment_unit_input.addItems(["emu", "A*m**2"])
        self.moment_unit_input.setFixedWidth(70)
        self.moment_target_unit_input = qtw.QComboBox()
        self.moment_target_unit_input.addItems(["emu", "A*m**2"])
        self.moment_target_unit_input.setFixedWidth(70)
        self.moment_label = qtw.QLabel("N/A")
        self.moment_target_label = qtw.QLabel(self.moment_target_unit_input.currentText())
        moment_info = qtw.QToolButton()
        moment_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        moment_info.setToolTip("Specify the original and target units for moment (e.g. 'emu' to 'A*m**2').")

        self.moment_target_unit_input.currentIndexChanged.connect(
            lambda: self.moment_target_label.setText(self.moment_target_unit_input.currentText())
        )

        self.moment_unit_input.currentIndexChanged.connect(self.reprocess_data)
        self.moment_target_unit_input.currentIndexChanged.connect(self.reprocess_data)

        # Arrange: [original unit] → [target unit] [=] [converted label] [info]
        moment_layout.addWidget(qtw.QLabel("Mom:"))
        moment_layout.addWidget(self.moment_unit_input)
        moment_layout.addWidget(qtw.QLabel("→"))
        moment_layout.addWidget(self.moment_target_unit_input)
        moment_layout.addWidget(qtw.QLabel("="))
        moment_layout.addWidget(self.moment_label)
        moment_layout.addWidget(self.moment_target_label)
        moment_layout.addWidget(moment_info)
        controls.addLayout(moment_layout)

        # ---------------------------- Regarding splitting branches ---------------------------- #
        # Branch splitting method label with info icon
        method_layout = qtw.QHBoxLayout()
        method_label = qtw.QLabel("Branch splitting method")
        method_layout.addWidget(method_label)

        info_button = qtw.QToolButton()
        info_button.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        info_button.setToolTip(
            "Select the method used for splitting branches in the analysis.\n"
            "'Diff': Separates based on the increase or decrease of the field.\n"
            "'Zero': Uses the zero-crossing method. Takes the upper branch as the first half of the data and the lower branch as the second half.\n"
            "The method can affect the interpretation of the hysteresis loop.")
        method_layout.addWidget(info_button)
        method_layout.addStretch()
        controls.addLayout(method_layout)
        # Method dropdown
        self.method_dropdown = qtw.QComboBox()
        self.method_dropdown.addItems(["Diff", "Zero"])
        self.method_dropdown.setToolTip("Choose 'Diff' for the difference method or 'Zero' for the zero-crossing method.")
        controls.addWidget(self.method_dropdown)
        self.method_dropdown.currentIndexChanged.connect(self.reprocess_data)

        # Add Reset All button at the bottom of controls
        self.reset_button = qtw.QPushButton("Reset All")
        self.reset_button.setToolTip("Reset all fields, selections, and plots to default.")
        self.reset_button.clicked.connect(self.reset_all)
        controls.addWidget(self.reset_button)

        # ------------------------------- Regarding corrections -------------------------------- #
        # Center shift checkbox
        center_layout = qtw.QHBoxLayout()
        self.center_shift_checkbox = qtw.QCheckBox("Center Shift")
        center_info = qtw.QToolButton()
        center_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        center_info.setToolTip("Check this box to apply center shift correction to the data.\nThis will adjust the horizontal and vertical offsets to center the hysteresis loop.\nIf the checkbox is unchecked, the center shift will not be applied, but the data will still be gridded.")
        center_layout.addWidget(self.center_shift_checkbox)
        center_layout.addWidget(center_info)
        center_layout.addStretch()
        corrections.addLayout(center_layout)

        # Drift correction dropdown
        self.drift_correction_dropdown = qtw.QComboBox()
        self.drift_correction_dropdown.addItems(["None", "Auto", "Symmetric", "Positive", "Upper"])
        drift_layout = qtw.QHBoxLayout()
        drift_label = qtw.QLabel("Drift Correction:")
        drift_layout.addWidget(drift_label)
        drift_layout.addWidget(self.drift_correction_dropdown)
        drift_info = qtw.QToolButton()
        drift_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        drift_info.setToolTip("If 'Auto' is selected, the program will determine the best method based on the data.\n"
                              "If 'Symmetric' is selected, symmetric averaging with tip-to-tip closure is applied.\n"
                              "If 'Positive' is selected, positive field correction is applied.\n"
                              "If 'Upper' is selected, the upper branch is used for correction. While this guarantees loop centering, it may introduce artificial symmetry and should therefore be used with caution.\n"
                              "If 'None' is selected, no drift correction is applied.\n")
        drift_layout.addWidget(drift_info)
        drift_layout.addStretch()
        corrections.addLayout(drift_layout)

        # Slope saturation checkbox
        #slope_layout = qtw.QHBoxLayout()
        #self.slope_correction_checkbox = qtw.QCheckBox("Slope Correction")
        #slope_info = qtw.QToolButton()
        #slope_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        #slope_info.setToolTip("Apply slope correction to the data based on the approach to saturation model.\nThis method is to be used ONLY when the ferromagnetic material is saturated and the linear contribution is mainly due to Dia or Para magnetism.\n")
        #slope_layout.addWidget(self.slope_correction_checkbox)
        #slope_layout.addWidget(slope_info)
        #slope_layout.addStretch()
        #corrections.addLayout(slope_layout)

        # Slope correction dropdown
        self.slope_correction_dropdown = qtw.QComboBox()
        self.slope_correction_dropdown.addItems(["None", "Linear", "Approach"])  # Add more methods as needed
        slope_layout = qtw.QHBoxLayout()
        slope_layout.addWidget(qtw.QLabel("Slope Correction:"))
        slope_layout.addWidget(self.slope_correction_dropdown)
        slope_info = qtw.QToolButton()
        slope_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        slope_info.setToolTip(
            "Select the slope correction method:\n"
            "'None': No correction.\n"
            "'Linear': Fit and subtract a linear function above 80% saturation.\n"
            "'Approach': Use the approach to saturation model (Jackson & Solheid Eq. 18).\n"
        )
        slope_layout.addWidget(slope_info)
        slope_layout.addStretch()
        corrections.addLayout(slope_layout)

        # Corrections Button
        self.corrections_button = qtw.QPushButton("Apply Corrections")
        self.corrections_button.clicked.connect(self.apply_corrections)
        corrections.addWidget(self.corrections_button)
        self.corrections_button.setToolTip("Apply the selected corrections to the data.\n"
                                           "Center shift will adjust the horizontal and vertical offsets.\n"
                                           "Drift correction will apply the selected method.\n"
                                           "Slope correction will apply the slope saturation model.")
        
        # ------------------------------- Regarding fitting ------------------------------------ #
        # Fitting dropdown
        fit_layout = qtw.QVBoxLayout()
        fit_mode_row = qtw.QHBoxLayout()
        self.fit_mode_dropdown = qtw.QComboBox()
        self.fit_mode_dropdown.addItems(["Single", "Multi", "Duhalde", "Custom"])
        fit_info = qtw.QToolButton()
        fit_info.setIcon(self.style().standardIcon(qtw.QStyle.SP_MessageBoxInformation))
        fit_info.setToolTip(
            "Single: Fits Mih/Mrh with a single function (logistic/tanh).\n"
            "Multi: Uses a basis set for more flexible fitting.\n"
            "Duhalde: Uses Duhalde's arctan model.\n"
            "Custom: User provides a function and initial guesses."
        )
        fit_mode_row.addWidget(qtw.QLabel("Fitting mode:"))
        fit_mode_row.addWidget(self.fit_mode_dropdown)
        fit_mode_row.addWidget(fit_info)
        fit_layout.addLayout(fit_mode_row)

        # Duhalde fitting parameter guesses
        self.Duhalde_guess_input = qtw.QLineEdit()
        self.Duhalde_guess_input.setPlaceholderText("Duhalde initial guesses (e.g. 1.0, 0.1, 0.01) in this order: Hc, Mr, Chi")

        # Upper branch function and guesses
        self.custom_func_up_input = qtw.QLineEdit()
        self.custom_func_up_input.setPlaceholderText("Upper branch function (e.g. lambda x,a,b: a*np.tanh(b*x))")
        self.custom_guess_up_input = qtw.QLineEdit()
        self.custom_guess_up_input.setPlaceholderText("Upper branch initial guesses (comma-separated, e.g. 1.0, 0.1)")

        # Lower branch function and guesses
        self.custom_func_lo_input = qtw.QLineEdit()
        self.custom_func_lo_input.setPlaceholderText("Lower branch function (e.g. lambda x,a,b: a*np.tanh(b*x))")
        self.custom_guess_lo_input = qtw.QLineEdit()
        self.custom_guess_lo_input.setPlaceholderText("Lower branch initial guesses (comma-separated, e.g. 1.0, 0.1)")

        fit_layout.addWidget(self.custom_func_up_input)
        fit_layout.addWidget(self.custom_guess_up_input)
        fit_layout.addWidget(self.custom_func_lo_input)
        fit_layout.addWidget(self.custom_guess_lo_input)
        fit_layout.addWidget(self.Duhalde_guess_input)

        self.fit_button = qtw.QPushButton("Fit Data")
        self.fit_button.setToolTip("Run the selected fitting method on the current data.")
        self.fit_button.clicked.connect(self.run_fitting)
        fit_layout.addWidget(self.fit_button)

        # Parameter display
        self.param_display = qtw.QTextEdit()
        self.param_display.setReadOnly(True)
        self.param_display.setFixedHeight(80)
        fit_layout.addWidget(self.param_display)

        # ---------------------------- Regarding exporting ------------------------------------- #
        self.export_button = qtw.QPushButton("Export Results")
        self.export_button.setToolTip("Export sample info, units, fit plot, and extracted parameters.")
        self.export_button.clicked.connect(self.export_results)
        fit_layout.addWidget(self.export_button)

        # Add fit_layout to your main controls or grid as appropriate
        fittings.addLayout(fit_layout)

        # ---------------------------- Regarding plotting -------------------------------------- #
        # Plot 1
        self.figure1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        # Plot 2
        self.figure2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        # Plot 3
        self.figure3, self.ax3 = plt.subplots()
        self.canvas3 = FigureCanvas(self.figure3)
        self.toolbar3 = NavigationToolbar(self.canvas3, self)

        # ---------------------------- Regarding gridlayout ------------------------------------ #
        # Gridding the layout
        grid.addLayout(controls, 2, 1, 1, 1)  # Controls in the first column
        grid.addWidget(self.toolbar1, 0, 1)
        grid.addWidget(self.canvas1, 1, 1)
        grid.addWidget(self.toolbar2, 0, 2)
        grid.addWidget(self.canvas2, 1, 2)
        grid.addWidget(self.toolbar3, 0, 3)
        grid.addWidget(self.canvas3, 1, 3)
        grid.addLayout(corrections, 2, 2, 1, 1)
        grid.addLayout(fittings, 2, 3, 1, 1)

        self.data_instance = None  # Will hold the Read_Data instance
        self.show()

    # Function to load the file and process data
    def load_file(self):
        # Overwrite warning if data already loaded
        if self.data_instance is not None:
            reply = qtw.QMessageBox.question(
                self,
                "Overwrite Data?",
                "Loading a new file will overwrite the current data and reset all settings. Continue?",
                qtw.QMessageBox.Yes | qtw.QMessageBox.No,
                qtw.QMessageBox.No
            )
            if reply != qtw.QMessageBox.Yes:
                return

        file_path, _ = qtw.QFileDialog.getOpenFileName(self, "Select .DAT or .CSV File", "", "Data Files (*.dat *.csv)")
        if not file_path:
            return

        # Extract the file name from the path
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        method = self.method_dropdown.currentText()

        # Read user input for volume and mass
        try:
            user_vol = float(self.vol_input.text()) if self.vol_input.text().strip() else None
        except ValueError:
            qtw.QMessageBox.warning(self, "Input Error", "Invalid volume. Please enter a number.")
            return

        try:
            user_mass = float(self.mass_input.text()) if self.mass_input.text().strip() else None
        except ValueError:
            qtw.QMessageBox.warning(self, "Input Error", "Invalid mass. Please enter a number.")
            return

        try:
            # Read user input for units
            vol_unit = self.vol_unit_input.currentText().strip() or None
            vol_target_unit = self.vol_target_unit_input.currentText().strip() or None
            mass_unit = self.mass_unit_input.currentText().strip() or None
            mass_target_unit = self.mass_target_unit_input.currentText().strip() or None
            # Read user input for field and magnetization units
            field_unit = self.field_unit_input.currentText().strip() or None
            field_target_unit = self.field_target_unit_input.currentText().strip() or None
            mom_unit = self.moment_unit_input.currentText().strip() or None
            mom_target_unit = self.moment_target_unit_input.currentText().strip() or None
        

            self.data_instance = Read_Data(file_path, method, 
                                           Volume=user_vol, Mass=user_mass,
                                           VolUnit=vol_unit, MassUnit=mass_unit,
                                           TargetVolUnit=vol_target_unit, TargetMassUnit=mass_target_unit,
                                           FieldUnit=field_unit, TargetFieldUnit=field_target_unit,
                                           MomUnit=mom_unit, TargetMomUnit=mom_target_unit)
            self.data_instance.sample_name = sample_name
            # Create auxiliary curves if available
            Auxiliary_Curves(self.data_instance)
            self.data_instance.file_path = file_path  # Store file_path for future reprocessing
            # Update labels based on target units or original units
            if self.field_target_unit_input.currentText().strip():
                self.field_x_label ="Field"+"("+self.field_target_unit_input.currentText().strip()+")"
            else:
                self.field_x_label = self.data_instance.FieldHead
                
            if user_vol is not None:
                self.mag_y_label = "Magnetization"+"("+self.moment_target_unit_input.currentText().strip()+'/'+self.vol_target_unit_input.currentText().strip()+")"
            elif user_mass is not None:
                self.mag_y_label = "Magnetization"+"("+self.moment_target_unit_input.currentText().strip()+'/'+self.mass_target_unit_input.currentText().strip()+")"
            else:
                self.mag_y_label = self.data_instance.MagHead

        except Exception as e:
            qtw.QMessageBox.critical(self, "Read Error", f"Could not read data:\n{str(e)}")
            return

        # Update metadata display
        self.vol_label.setText(f"{self.data_instance.Vol}" if self.data_instance.Vol else "N/A")
        self.mass_label.setText(f"{self.data_instance.Mass}" if self.data_instance.Mass else "N/A")
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        # Plot data
        self.Plot_Branches()
        self.Plot_Auxiliary_Curves()


    # Function to reprocess data based on user input
    def reprocess_data(self):
        # Only reprocess if a file has already been loaded
        if not self.data_instance:
            return

        file_path = self.data_instance.file_path if hasattr(self.data_instance, 'file_path') else None
        if not file_path:
            qtw.QMessageBox.warning(self, "No File", "No file loaded to reprocess.")
            return

        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        method = self.method_dropdown.currentText()

        try:
            user_vol = float(self.vol_input.text()) if self.vol_input.text().strip() else None
        except ValueError:
            qtw.QMessageBox.warning(self, "Input Error", "Invalid volume. Please enter a number.")
            return

        try:
            user_mass = float(self.mass_input.text()) if self.mass_input.text().strip() else None
        except ValueError:
            qtw.QMessageBox.warning(self, "Input Error", "Invalid mass. Please enter a number.")
            return

        try:
            # Read user input for units
            vol_unit = self.vol_unit_input.currentText().strip() or None
            vol_target_unit = self.vol_target_unit_input.currentText().strip() or None
            mass_unit = self.mass_unit_input.currentText().strip() or None
            mass_target_unit = self.mass_target_unit_input.currentText().strip() or None
            # Read user input for field and magnetization units
            field_unit = self.field_unit_input.currentText().strip() or None
            field_target_unit = self.field_target_unit_input.currentText().strip() or None
            mom_unit = self.moment_unit_input.currentText().strip() or None
            mom_target_unit = self.moment_target_unit_input.currentText().strip() or None
     
            self.data_instance = Read_Data(file_path, method, 
                                           Volume=user_vol, Mass=user_mass,
                                           VolUnit=vol_unit, MassUnit=mass_unit,
                                           TargetVolUnit=vol_target_unit, TargetMassUnit=mass_target_unit,
                                           FieldUnit=field_unit, TargetFieldUnit=field_target_unit,
                                           MomUnit=mom_unit, TargetMomUnit=mom_target_unit)
            self.data_instance.sample_name = sample_name
            # Create auxiliary curves if available
            Auxiliary_Curves(self.data_instance)
            self.data_instance.file_path = file_path  # Store file_path for future reprocessing
            # Update labels based on target units or original units
            if self.field_target_unit_input.currentText().strip():
                self.field_x_label ="Field"+"("+self.field_target_unit_input.currentText().strip()+")"
            else:
                self.field_x_label = self.data_instance.FieldHead
                
            if user_vol is not None:
                self.mag_y_label = "Magnetization"+"("+self.moment_target_unit_input.currentText().strip()+'/'+self.vol_target_unit_input.currentText().strip()+")"
            elif user_mass is not None:
                self.mag_y_label = "Magnetization"+"("+self.moment_target_unit_input.currentText().strip()+'/'+self.mass_target_unit_input.currentText().strip()+")"
            else:
                self.mag_y_label = self.data_instance.MagHead

        except Exception as e:
            qtw.QMessageBox.critical(self, "Read Error", f"Could not read data:\n{str(e)}")
            return

        self.vol_label.setText(f"{self.data_instance.Vol}" if self.data_instance.Vol else "N/A")
        self.mass_label.setText(f"{self.data_instance.Mass}" if self.data_instance.Mass else "N/A")
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        # Plot data
        self.Plot_Branches()
        self.Plot_Auxiliary_Curves()

    def apply_corrections(self):
        if not self.data_instance:
            qtw.QMessageBox.warning(self, "No Data", "No data loaded to apply corrections.")
            return

        # Apply center shift correction
        if self.center_shift_checkbox.isChecked():
            Center_Shift(self.data_instance, Apply=True)
        elif self.center_shift_checkbox.isChecked() == False:
            Center_Shift(self.data_instance, Apply=False)
        
        # Apply drift correction
        drift_method = self.drift_correction_dropdown.currentText()
        if drift_method != "None":
            Drift_Correction(self.data_instance, Mode=drift_method)

        # Apply slope correction
        slope_method = self.slope_correction_dropdown.currentText()
        if slope_method != "None":
            Slope_Correction(self.data_instance, method=slope_method)
        #if self.slope_correction_checkbox.isChecked():
        #    Slope_Correction(self.data_instance, Apply=True)

        self.Plot_Corrections()

    def run_fitting(self):
        if not self.data_instance:
            qtw.QMessageBox.warning(self, "No Data", "No data loaded to fit.")
            return

        fit_mode = self.fit_mode_dropdown.currentText()
        func_up = self.custom_func_up_input.text()
        guess_up = [float(val.strip()) for val in self.custom_guess_up_input.text().split(',')] if self.custom_guess_up_input.text() else None
        func_lo = self.custom_func_lo_input.text()
        guess_lo = [float(val.strip()) for val in self.custom_guess_lo_input.text().split(',')] if self.custom_guess_lo_input.text() else None
        guess_Duhalde = [float(val.strip()) for val in self.Duhalde_guess_input.text().split(',')] if self.Duhalde_guess_input.text() else None

        try:
            if fit_mode == "Duhalde":
                # Instantiate the fitting class
                self.fitting_instance = Fitting(
                    self.data_instance,
                    mode=fit_mode,
                    guess_Duhalde=guess_Duhalde)
            elif fit_mode == "Custom":
                # Instantiate the fitting class with custom functions and guesses
                self.fitting_instance = Fitting(
                    self.data_instance,
                    mode=fit_mode,
                    func_up=func_up,
                    guess_up=guess_up,
                    func_lo=func_lo,
                    guess_lo=guess_lo)
            else:
                # Instantiate the fitting class with the selected mode
                self.fitting_instance = Fitting(self.data_instance, mode=fit_mode)
        except ValueError:
            qtw.QMessageBox.warning(self, "Input Error", "Please check your inputs.")
            return

        # --- Gather all fitting info ---
        fit_params_text = ""  
        # 0. Optimal parameters from curve_fit
        if fit_mode == "Custom":
            params_up = getattr(self.fitting_instance, 'opt_params_up', None)
            params_lo = getattr(self.fitting_instance, 'opt_params_lo', None)
            if params_up is not None and params_lo is not None:
                fit_params_text = (
                    "Upper branch fit parameters:\n" +
                    ", ".join([f"{v:.4g}" for v in params_up]) +
                    "\nLower branch fit parameters:\n" +
                    ", ".join([f"{v:.4g}" for v in params_lo]) +
                    "\n")

        # 1. Physical parameters from Parameter_Extraction
        params = getattr(self.fitting_instance.Sam, 'Params', {})
        moment_unit = self.moment_target_unit_input.currentText().strip()
        vol_unit = self.vol_target_unit_input.currentText().strip()
        mass_unit = self.mass_target_unit_input.currentText().strip()
        field_unit = self.field_target_unit_input.currentText().strip()
        # Decide denominator for magnetization
        if self.vol_input.text().strip():
            mag_unit = f"{moment_unit}/{vol_unit}"
        elif self.mass_input.text().strip():
            mag_unit = f"{moment_unit}/{mass_unit}"
        else:
            mag_unit = moment_unit  # fallback

        if params:
            fit_params_text += "\nPhysical parameters:\n"
            for k, v in params.items():
                if k in ("Mr_Up", "Mr_Lo", "Ms_Up", "Ms_Lo"):
                    fit_params_text += f"{k}: {v:.4g} {mag_unit}\n"
                elif k == "Hc_Up" or k == "Hc_Lo":
                    fit_params_text += f"{k}: {v:.4g} {field_unit}\n"
                else:
                    fit_params_text += f"{k}: {v:.4g}\n"
        else:
            fit_params_text += "\nNo physical parameters extracted.\n"
        
        # 2. Metrics (display the whole dictionary)
        metrics = getattr(self.fitting_instance.Sam, 'Metrics', {})
        if metrics:
            fit_params_text += "\nMetrics:\n"
            for k, v in metrics.items():
                fit_params_text += f"{k}: {v}\n"
        # display the fit parameters in the text box
        self.param_display.setText(fit_params_text)
        # Plot the fitting results
        self.Plot_Fitting()

    def Plot_Branches(self):
        if not self.data_instance:
            qtw.QMessageBox.warning(self, "No Data", "No data loaded to plot branches.")
            return

        self.ax1.clear()
        Up_field = self.data_instance.Up[self.data_instance.FieldHead]
        Up_mag = self.data_instance.Up[self.data_instance.MagHead]
        Lo_field = self.data_instance.Lo[self.data_instance.FieldHead]
        Lo_mag = self.data_instance.Lo[self.data_instance.MagHead]
        self.ax1.scatter(Up_field, Up_mag, label='Upper branch', color='blue', s=3)
        self.ax1.scatter(Lo_field, Lo_mag, label='Lower branch', color='red',  s=3)
        self.ax1.set_xlabel(self.field_x_label)
        self.ax1.set_ylabel(self.mag_y_label)
        self.ax1.set_title(f"Hysteresis Loop for {self.data_instance.sample_name}")
        self.ax1.legend()
        # Add major and minor ticks
        self.ax1.xaxis.set_major_locator(mticker.AutoLocator())
        self.ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        self.ax1.yaxis.set_major_locator(mticker.AutoLocator())
        self.ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        self.ax1.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add a vertical line at zero field and horizontal line at zero magnetization
        self.ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        self.ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        # Draw the canvas
        self.canvas1.draw()

    def Plot_Auxiliary_Curves(self):
        if not self.data_instance or not hasattr(self.data_instance, 'Aux'):
            qtw.QMessageBox.warning(self, "No Auxiliary Data", "No auxiliary curves available to plot.")
            return

        self.ax2.clear()
        Aux = self.data_instance.Aux
        self.ax2.scatter(Aux[self.data_instance.FieldHead], Aux['Mih'], label='Induced Hysteretic Magnetization (Mih)', color='green', s=3)
        self.ax2.scatter(Aux[self.data_instance.FieldHead], Aux['Mrh'], label='Remanent Hysteretic Magnetization (Mrh)', color='orange', s=3)
        self.ax2.set_xlabel(self.field_x_label)
        self.ax2.set_ylabel(self.mag_y_label)
        self.ax2.set_title(f"Auxiliary Curves for {self.data_instance.sample_name}")
        self.ax2.legend()
        # Add major and minor ticks
        self.ax2.xaxis.set_major_locator(mticker.AutoLocator())
        self.ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        self.ax2.yaxis.set_major_locator(mticker.AutoLocator())
        self.ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        self.ax2.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add a vertical line at zero field and horizontal line at zero magnetization
        self.ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        self.ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        # Draw the canvas
        self.canvas2.draw()

    def Plot_Corrections(self):
        if not self.data_instance:
            qtw.QMessageBox.warning(self, "No Data", "No data loaded to plot corrections.")
            return
        
        Up_field = self.data_instance.Up[self.data_instance.FieldHead]
        Up_mag = self.data_instance.Up[self.data_instance.MagHead]
        Lo_field = self.data_instance.Lo[self.data_instance.FieldHead]
        Lo_mag = self.data_instance.Lo[self.data_instance.MagHead]
        self.ax1.scatter(Up_field, Up_mag, label='Corrected Upper branch', color='cyan', s=3)
        self.ax1.scatter(Lo_field, Lo_mag, label='Corrected Lower branch', color='orange',  s=3)
        self.ax1.set_xlabel(self.field_x_label)
        self.ax1.set_ylabel(self.mag_y_label)
        self.ax1.legend()
        self.canvas1.draw()

        # Re-plot auxiliary curves after corrections
        Aux = self.data_instance.Aux
        self.ax2.scatter(Aux[self.data_instance.FieldHead], Aux['Mih'], label='Corrected Mih', color='aqua', s= 6)  
        self.ax2.scatter(Aux[self.data_instance.FieldHead], Aux['Mrh'], label='Corrected Mrh', color='magenta', s= 6)
        self.ax2.set_xlabel(self.field_x_label)
        self.ax2.set_ylabel(self.mag_y_label)
        self.ax2.legend()
        self.canvas2.draw()

    def Plot_Fitting(self):
        if not hasattr(self, 'fitting_instance'):
            qtw.QMessageBox.warning(self, "No Fitting Data", "No fitting data available to plot.")
            return

        self.ax3.clear()
        # Plot the fitted curves
        self.ax3.scatter(self.fitting_instance.Sam.Up[self.data_instance.FieldHead], 
                         self.fitting_instance.Sam.Up[self.data_instance.MagHead],
                         label='Upper Branch', color='cyan', s=3)
        self.ax3.scatter(self.fitting_instance.Sam.Lo[self.data_instance.FieldHead],
                         self.fitting_instance.Sam.Lo[self.data_instance.MagHead],
                         label='Lower Branch', color='orange', s=3)
        self.ax3.plot(self.fitting_instance.Sam.Up[self.data_instance.FieldHead],
                      self.fitting_instance.Sam.Up['Fit'],
                      label='Fitted Upper Branch', color='blue', linewidth=1.5)
        self.ax3.plot(self.fitting_instance.Sam.Lo[self.data_instance.FieldHead],
                      self.fitting_instance.Sam.Lo['Fit'],
                      label='Fitted Lower Branch', color='red', linewidth=1.5)

        self.ax3.set_xlabel(self.field_x_label)
        self.ax3.set_ylabel(self.mag_y_label)
        self.ax3.set_title(f"Fitting Results for {self.data_instance.sample_name}")
        self.ax3.legend()
        # Add major and minor ticks
        self.ax3.xaxis.set_major_locator(mticker.AutoLocator())
        self.ax3.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        self.ax3.yaxis.set_major_locator(mticker.AutoLocator())
        self.ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        self.ax3.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add a vertical line at zero field and horizontal line at zero magnetization
        self.ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        self.ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        # Draw the canvas
        self.canvas3.draw()


    def export_results(self):
        if not self.data_instance:
            qtw.QMessageBox.warning(self, "No Data", "No data loaded to export.")
            return

        # Ask user for export folder
        folder = qtw.QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return

        sample_name = getattr(self.data_instance, 'sample_name', 'Unknown')
        vol = getattr(self.data_instance, 'Vol', 'N/A')
        mass = getattr(self.data_instance, 'Mass', 'N/A')
        vol_unit = self.vol_unit_input.currentText()
        mass_unit = self.mass_unit_input.currentText()
        field_unit = self.field_unit_input.currentText()
        moment_unit = self.moment_unit_input.currentText()
        field_target_unit = self.field_target_unit_input.currentText()
        moment_target_unit = self.moment_target_unit_input.currentText()

        params = getattr(getattr(self, 'fitting_instance', None), 'Sam', None)
        param_dict = getattr(params, 'Params', {})
        metrics_dict = getattr(params, 'Metrics', {})

        # --- Add fit parameters for custom function ---
        fit_mode = self.fit_mode_dropdown.currentText()

        # --- Corrections applied ---
        corrections_applied = []
        if self.center_shift_checkbox.isChecked():
            corrections_applied.append("Center Shift")
        drift_method = self.drift_correction_dropdown.currentText()
        if drift_method != "None":
            corrections_applied.append(f"Drift Correction ({drift_method})")
        if self.slope_correction_checkbox.isChecked():
            corrections_applied.append("Slope Correction")
        if not corrections_applied:
            corrections_applied.append("None")

        # --- Add fit parameters for custom function ---
        fit_mode = self.fit_mode_dropdown.currentText()
        txt_lines = [
            f"Sample Name: {sample_name}",
            f"Volume: {vol} {vol_unit}",
            f"Mass: {mass} {mass_unit}",
            f"Field Unit: {field_unit} (target: {field_target_unit})",
            f"Moment Unit: {moment_unit} (target: {moment_target_unit})",
            "",
            "Corrections Applied:",
            ", ".join(corrections_applied),
            "",
            "Fitting Method:",
            f"Mode: {fit_mode}"
        ]

        # If custom, include function and guesses
        if fit_mode == "Custom":
            txt_lines.append(f"Upper branch function: {self.custom_func_up_input.text()}")
            txt_lines.append(f"Upper branch initial guesses: {self.custom_guess_up_input.text()}")
            txt_lines.append(f"Lower branch function: {self.custom_func_lo_input.text()}")
            txt_lines.append(f"Lower branch initial guesses: {self.custom_guess_lo_input.text()}")

        txt_lines.append("")

        if fit_mode == "Custom":
            params_up = getattr(self.fitting_instance, 'opt_params_up', None)
            params_lo = getattr(self.fitting_instance, 'opt_params_lo', None)
            if params_up is not None and params_lo is not None:
                txt_lines.append("Upper branch fit parameters:")
                txt_lines.append(", ".join([f"{v:.4g}" for v in params_up]))
                txt_lines.append("Lower branch fit parameters:")
                txt_lines.append(", ".join([f"{v:.4g}" for v in params_lo]))
                txt_lines.append("Ignore phyisical parameters for custom fit.")
                txt_lines.append("The ones above are the fit parameters.")
            else:
                txt_lines.append("Fit parameters: None extracted.")

        txt_lines += [
            "",
            "Physical Parameters:"
        ]
        # Decide denominator for magnetization
        moment_target_unit = self.moment_target_unit_input.currentText()
        vol_target_unit = self.vol_target_unit_input.currentText()
        mass_target_unit = self.mass_target_unit_input.currentText()
        field_target_unit = self.field_target_unit_input.currentText()
        if self.vol_input.text().strip():
            mag_unit = f"{moment_target_unit}/{vol_target_unit}"
        elif self.mass_input.text().strip():
            mag_unit = f"{moment_target_unit}/{mass_target_unit}"
        else:
            mag_unit = moment_target_unit  # fallback

        if param_dict:
            for k, v in param_dict.items():
                if k in ("Mr_Up", "Mr_Lo", "Ms_Up", "Ms_Lo"):
                    txt_lines.append(f"{k}: {v:.4g} {mag_unit}")
                elif k in ("Hc_Up", "Hc_Lo"):
                    txt_lines.append(f"{k}: {v:.4g} {field_target_unit}")
                else:
                    txt_lines.append(f"{k}: {v:.4g}")
        else:
            txt_lines.append("None extracted.")

        txt_lines.append("\nMetrics:")
        if metrics_dict:
            txt_lines += [f"{k}: {v}" for k, v in metrics_dict.items()]
        else:
            txt_lines.append("None extracted.")

        # Export to PDF
        pdf_path = os.path.join(folder, f"{sample_name}_results.pdf")
        with PdfPages(pdf_path) as pdf:
            # 1. Add the fit plot
            pdf.savefig(self.figure3)

            # 2. Add a page with the text info
            fig_txt = plt.figure(figsize=(8.27, 11.69))  # A4 size
            plt.axis('off')
            plt.text(0, 1, '\n'.join(txt_lines), fontsize=12, va='top', family='monospace')
            pdf.savefig(fig_txt)
            plt.close(fig_txt)
    
            # --- Export corrected data as CSV ---
        try:
            data_df = getattr(self.data_instance, 'Data', None)
            if data_df is not None:
                csv_path = os.path.join(folder, f"{sample_name}_corrected_data.csv")
                data_df.to_csv(csv_path, index=False)
        except Exception as e:
            qtw.QMessageBox.warning(self, "Export Warning", f"Could not export corrected data CSV:\n{e}")


        qtw.QMessageBox.information(self, "Export Complete", f"Results exported to:\n{pdf_path}")

    def reset_all(self):
        # Reset text inputs
        self.vol_input.clear()
        self.mass_input.clear()
        self.custom_func_up_input.clear()
        self.custom_guess_up_input.clear()
        self.custom_func_lo_input.clear()
        self.custom_guess_lo_input.clear()
        self.Duhalde_guess_input.clear()

        # Reset dropdowns to first item
        self.vol_unit_input.setCurrentIndex(0)
        self.vol_target_unit_input.setCurrentIndex(0)
        self.mass_unit_input.setCurrentIndex(0)
        self.mass_target_unit_input.setCurrentIndex(0)
        self.field_unit_input.setCurrentIndex(0)
        self.field_target_unit_input.setCurrentIndex(0)
        self.moment_unit_input.setCurrentIndex(0)
        self.moment_target_unit_input.setCurrentIndex(0)
        self.method_dropdown.setCurrentIndex(0)
        self.fit_mode_dropdown.setCurrentIndex(0)
        self.drift_correction_dropdown.setCurrentIndex(0)
        self.slope_correction_dropdown.setCurrentIndex(0)

        # Reset checkboxes
        self.center_shift_checkbox.setChecked(False)
        
        # Reset labels
        self.vol_label.setText("N/A")
        self.mass_label.setText("N/A")
        self.field_label.setText("N/A")
        self.moment_label.setText("N/A")
        self.vol_target_label.setText(self.vol_target_unit_input.currentText())
        self.mass_target_label.setText(self.mass_target_unit_input.currentText())
        self.field_target_label.setText(self.field_target_unit_input.currentText())
        self.moment_target_label.setText(self.moment_target_unit_input.currentText())

        # Clear parameter display
        self.param_display.clear()

        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        # Reset data instance and fitting instance
        self.data_instance = None
        if hasattr(self, 'fitting_instance'):
            del self.fitting_instance

if __name__ == "__main__":
    app = qtw.QApplication([])
    font = qtg.QFont()
    font.setPointSize(8)
    app.setFont(font)
    mw = PyHystGUI()
    app.exec_()

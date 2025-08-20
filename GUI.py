
__version__ = "0.1"

import sys
import numpy as np
import scipy.io as sio
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from qtrangeslider._labeled import QLabeledDoubleRangeSlider
from common import get_file, split_cycles, select_from_collection, reset_alpha, draw_interpolant, reduce_data
from LM_Chaboche import *


class standalone_app(QtWidgets.QMainWindow):
    
    def __init__(self, parent=None):
        super(standalone_app, self).__init__(parent)
        self.main_window = interactor(self)
        self.setWindowTitle("LM_Chaboche GUI v%s" %(__version__))
        self.setCentralWidget(self.main_window)
        screen = QtWidgets.QApplication.primaryScreen()
        rect = screen.availableGeometry()
        self.setMinimumSize(QtCore.QSize(int(2*rect.width()/3), int(2*rect.height()/3)))

class main_window(QtWidgets.QWidget):
    '''
    Sets up main UI elements
    '''
    
    def setup(self,parent):
        
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        size_policy.setHorizontalStretch(50)
        size_policy.setVerticalStretch(50)
        self.setSizePolicy(size_policy)
        
        main_layout = QtWidgets.QHBoxLayout(parent)
        io_layout = QtWidgets.QVBoxLayout()
        
        self.load_button = QtWidgets.QPushButton("Load")
        
        self.offset_sb = QtWidgets.QDoubleSpinBox()
        self.offset_sb.setDecimals(0)
        self.offset_sb.setPrefix('Rp = ')
        self.offset_sb.setSuffix(' \u03BC\u03B5')
        self.offset_sb.setMinimum(1)
        self.offset_sb.setValue(20)
        self.offset_sb.setMaximum(2000)
        self.offset_sb.setEnabled(True)

        
        self.modulus_sb = QtWidgets.QDoubleSpinBox()
        self.modulus_sb.setDecimals(2)
        self.modulus_sb.setPrefix('E = ')
        self.modulus_sb.setSuffix(' GPa')
        self.modulus_sb.setMaximum(1000)
        self.modulus_sb.setEnabled(False)
        
        self.ys_sb = QtWidgets.QDoubleSpinBox()
        self.ys_sb.setDecimals(2)
        self.ys_sb.setPrefix('\u03C3y = ')
        self.ys_sb.setSuffix(' MPa')
        self.ys_sb.setMaximum(10000)
        self.ys_sb.setEnabled(False)
        
        
        vis_box = QtWidgets.QGroupBox("Initialization")
        vis_layout = QtWidgets.QGridLayout()
        
        vis_layout.addWidget(self.load_button,0,0,1,1)
        vis_layout.addWidget(self.offset_sb,1,0,1,1)
        vis_layout.addWidget(self.modulus_sb,1,1,1,1)
        vis_layout.addWidget(self.ys_sb,1,2,1,1)
        vis_box.setLayout(vis_layout)
        
        param_box = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QHBoxLayout()
        
        
        iso_box = QtWidgets.QGroupBox("Isotropic")
        iso_layout = QtWidgets.QGridLayout()
        self.iso_boundaries = make_boundary_button(self)
        iso_layout.addWidget(self.iso_boundaries,0,0,1,1)
        self.iso_input = {}
        iso_labels = ['Q','b']
        iso_values = [50, 5]
        i = 0
        for entry in iso_labels:
            self.iso_input[entry] = QtWidgets.QDoubleSpinBox()
            self.iso_input[entry].setPrefix('%s = '%iso_labels[i])
            self.iso_input[entry].setMaximum(1000)
            self.iso_input[entry].setMinimum(0)
            self.iso_input[entry].setDecimals(4)
            self.iso_input[entry].setValue(iso_values[i])
            iso_layout.addWidget(self.iso_input[entry],i+1,0,1,1)
            i+=1
        
        iso_box.setLayout(iso_layout)
        
        kin_box = QtWidgets.QGroupBox("Kinematic")
        kin_layout = QtWidgets.QGridLayout()
        self.kin_boundaries = make_boundary_button(self,'kinematic')
        kin_layout.addWidget(self.kin_boundaries, 0,0, 1, 1)
        self.C1_sb = QtWidgets.QDoubleSpinBox()
        self.C2_sb = QtWidgets.QDoubleSpinBox()
        self.C3_sb = QtWidgets.QDoubleSpinBox()
        self.C4_sb = QtWidgets.QDoubleSpinBox()
        self.r1_sb = QtWidgets.QDoubleSpinBox()
        self.r2_sb = QtWidgets.QDoubleSpinBox()
        self.r3_sb = QtWidgets.QDoubleSpinBox()
        self.r4_sb = QtWidgets.QDoubleSpinBox()
        self.kin_input = {}
        kin_labels = ['C1', 'C2', 'C3', 'C4', 'r1', 'r2', 'r3', 'r4']
        kin_text = ['C\u2081','C\u2082', 'C\u2083', 'C\u2084', 'r\u2081','r\u2082', 'r\u2083', 'r\u2084']
        kin_values = [50000, 10000, 5000, 0, 500, 100, 50, 0]
        i = 0
        for entry in kin_labels:
            self.kin_input[entry] = QtWidgets.QDoubleSpinBox()
            self.kin_input[entry].setPrefix('%s = '%kin_text[i])
            self.kin_input[entry].setMaximum(100000)
            self.kin_input[entry].setMinimum(-100000)
            self.kin_input[entry].setDecimals(4)
            self.kin_input[entry].setValue(kin_values[i])
            if i <= 3:
                kin_layout.addWidget(self.kin_input[entry],i+1,0,1,1)
            else:
                kin_layout.addWidget(self.kin_input[entry],i-3,1,1,1)
            i+=1
        kin_box.setLayout(kin_layout)
        
        run_layout = QtWidgets.QGridLayout()
        
        self.ignore_bounds_cb = QtWidgets.QCheckBox("Ignore bounds")
        self.reduce_data_sb = QtWidgets.QSpinBox()
        self.reduce_data_sb.setMaximum(100)
        self.reduce_data_sb.setValue(100)
        self.reduce_data_sb.setSuffix(' %')
        self.reduce_data_sb.setToolTip('Percentage of data to use in fitting')
        
        self.iter_sb = QtWidgets.QDoubleSpinBox()
        self.iter_sb.setPrefix('I = ')
        self.iter_sb.setToolTip('Maximum iterations')
        self.iter_sb.setMaximum(1000)
        self.iter_sb.setDecimals(0)
        self.iter_sb.setValue(100)
        
        self.damping_sb = QtWidgets.QDoubleSpinBox()
        self.damping_sb.setPrefix('\u03BB = ')
        self.damping_sb.setSuffix(' \u00D710\u207B\u2076')
        self.damping_sb.setToolTip('Damping factor')
        self.damping_sb.setMaximum(100)
        self.damping_sb.setDecimals(0)
        self.damping_sb.setValue(1)
        
        self.pbar = QtWidgets.QProgressBar(self, textVisible=True)
        self.pbar.setAlignment(Qt.AlignCenter)
        self.pbar.setFormat("Idle")
        self.pbar.setValue(0)
        self.run_button = QtWidgets.QPushButton('Run')
        
        run_layout.addWidget(self.ignore_bounds_cb,0,0,1,1)
        run_layout.addWidget(self.reduce_data_sb,0,1,1,1)
        run_layout.addWidget(self.iter_sb,0,2,1,1)
        run_layout.addWidget(self.damping_sb,0,3,1,1)
        run_layout.addWidget(self.pbar,1,0,1,3)
        run_layout.addWidget(self.run_button)
        
        param_layout.addWidget(iso_box)
        param_layout.addWidget(kin_box)
        fit_layout = QtWidgets.QVBoxLayout()
        fit_layout.addLayout(param_layout)
        fit_layout.addLayout(run_layout)

        param_box.setLayout(fit_layout)
        
        result_box = QtWidgets.QGroupBox("Results")
        result_layout = QtWidgets.QHBoxLayout()
        self.rsquared_le = QtWidgets.QLineEdit()
        self.rsquared_le.setText('R\u00B2: ')
        self.rmse_le = QtWidgets.QLineEdit()
        self.rmse_le.setText("RMSE: ")
        result_layout.addWidget(self.rsquared_le)
        result_layout.addWidget(self.rmse_le)
        result_box.setLayout(result_layout)
        
        io_layout.addWidget(vis_box)
        io_layout.addWidget(param_box)
        io_layout.addWidget(result_box)
        io_layout.addStretch()
        
        self.figure = plt.figure(figsize=(16,12))
        self.canvas = FigureCanvas(self.figure)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.canvas.setSizePolicy(size_policy)
        self.cycle_slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.cycle_slider.setRange(0, 1)
        self.cycle_slider.setSingleStep(0.25)#for quarter steps
        self.cycle_slider.setValue((0, 1))
        self.cycle_slider.setEnabled(False)
        self.update_plot_button = QtWidgets.QPushButton("Update")
        self.update_plot_button.setToolTip("Update the plot according to new cycles")
        self.draw_fit_button = QtWidgets.QPushButton("Show fit")
        self.draw_fit_button.setToolTip("Show fit according to parameters")
        
        toolbar = NavigationToolbar(self.canvas)
        
        plt_layout = QtWidgets.QVBoxLayout()
        plt_layout.addWidget(self.canvas)
        plt_layout.addStretch()
        plt_control_layout = QtWidgets.QHBoxLayout()
        v1 = QtWidgets.QVBoxLayout()
        plt_control_layout.addWidget(self.cycle_slider)
        v1.addWidget(self.update_plot_button)
        v1.addWidget(self.draw_fit_button)
        plt_control_layout.addLayout(v1)
        plt_layout.addLayout(plt_control_layout)
        plt_layout.addWidget(toolbar)
        
        main_layout.addLayout(io_layout)
        main_layout.addLayout(plt_layout)

class interactor(QtWidgets.QWidget):
    '''
    Connects and defines interactions between the main_window, inheriting most characteristics from a QWidget
    '''
    def __init__(self,parent):
        super(interactor, self).__init__(parent)
        self.ui = main_window()
        self.ui.setup(self)
        
        self.ui.load_button.clicked.connect(self.load_file)
        self.ui.update_plot_button.clicked.connect(self.update_plot)
        self.ui.run_button.clicked.connect(self.run)
        self.ui.draw_fit_button.clicked.connect(self.draw_chaboche)
        
    def cycle_value_change(self):
        #sanitize inputs, rounding to nearest 0.25
        def round_to_quarters(val):
            return round (val / 0.25) * 0.25
        value = self.ui.cycle_slider.value()
        self.ui.cycle_slider.setValue((round_to_quarters(value[0]), round_to_quarters(value[1])))
        

    def load_file(self):
        '''
        Loads a csv file
        '''
        fname = get_file("*.csv")
        
        if fname is None:
            return
            
        data = np.genfromtxt(fname, skip_header=1, delimiter=',')
        #break up data according to relevant columns
        #assumes incoming np array data has column 5 column being cycle, -2 is strain, -1 is stress
        self.alldata = data[:,[5,-2,-1]]
        self.cycle_ind, self.cycles = split_cycles(self.alldata, True)
        self.ui.cycle_slider.setRange(0, max(self.cycles))
        self.ui.cycle_slider.setValue((0, 0.25)) #onload
        if not self.ui.cycle_slider.isEnabled():
            self.ui.cycle_slider.setEnabled(True)
        self.update_plot()
    
    def update_plot(self):
        self.ui.figure.clear()
        ax = self.ui.figure.gca()
        if hasattr(self,'cycles'):
            self.cycle_value_change() #round values
            #find index corresponding to the slider 
            ind = np.nonzero((self.cycles>self.ui.cycle_slider.value()[0]) & (self.cycles<=self.ui.cycle_slider.value()[1]))
            ind = [int(x) for x in ind[0]] #make sure it can be used to index
            
            self.visible_data = self.alldata[self.cycle_ind[ind[0]][0]:self.cycle_ind[ind[-1]][1],1::]
        
        #replot everything

        ax.set_xlabel(r'Strain $\epsilon$ (%)', fontsize=12)
        ax.set_ylabel(r'Stress $\sigma$ (MPa)', fontsize=12)
        ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.1)
        ax.minorticks_on()
        ax.grid(visible=True, which='minor', color='#666666', linestyle='-', alpha=0.2)
        if hasattr(self,'visible_data'):
            self.pts = ax.scatter(self.visible_data[:,0],self.visible_data[:,1], s=1)
        else:
            return
        self.selector = select_from_collection(ax, self.pts, "lasso", 0.1)
        
        self.ui.canvas.draw()

    def draw_chaboche(self):
        if not hasattr(self,'visible_data'):
            return
        #remove existing lines
        try:
            for art in list(self.ui.figure.gca().lines):
                art.remove()
        except:
            pass

        param_dict = dict(self.ui.iso_input, **self.ui.kin_input)
        param_key = ['C1', 'r1', 'C2', 'r2', 'C3', 'r3', 'Q', 'b']
        params = []
        for key in param_key:
            params.append(param_dict[key].value())
        params = np.asarray(params)
        
        ind = np.nonzero((self.cycles<=self.ui.cycle_slider.value()[1]))
        ind = [int(x) for x in ind[0]]
        data = self.alldata[self.cycle_ind[ind[0]][0]:self.cycle_ind[ind[-1]][1],1::]
        
        self.predicted_stress = chaboche_model(
        data[:,0]/100,
        params, 
        self.ui.modulus_sb.value()*1000, 
        self.ui.ys_sb.value())
        
        ind = np.nonzero((self.cycles>self.ui.cycle_slider.value()[0]) & (self.cycles<=self.ui.cycle_slider.value()[1]))
        ind = [int(x) for x in ind[0]] #make sure it can be used to index
        chaboche_stress = self.predicted_stress[self.cycle_ind[ind[0]][0]:self.cycle_ind[ind[-1]][1]]
        
        ax = self.ui.figure.gca()
        ax.plot(self.visible_data[:,0],chaboche_stress, color='orange')
        self.ui.canvas.draw()
        self.update_error()

    def update_error(self):
        
        sse = np.sum((self.alldata[0:len(self.predicted_stress),2] - self.predicted_stress) ** 2)
        sst = np.sum((self.alldata[0:len(self.predicted_stress),2] - np.mean(self.alldata[0:len(self.predicted_stress),2])) ** 2)
        r_squared = 1 - (sse / sst)
        rmse = np.sqrt(np.mean((self.alldata[0:len(self.predicted_stress),2] - self.predicted_stress) ** 2))
        
        self.ui.rsquared_le.setText('R\u00B2: %f'%r_squared)
        self.ui.rmse_le.setText("RMSE: %f"%rmse)
        
    def keyPressEvent(self,event):
        key = event.key()
        if key == 16777220:#should be "QtCore.Qt.Key_Enter():#" but it's in use by other widgets
            try:
                for art in list(self.ui.figure.gca().lines):
                    art.remove()
            except:
                pass
            try:
                reset_alpha(self.ui.figure.gca(),self.pts)
            except:
                return #if reset_alpha fails, there is no data, and drawing an interpolate is difficult
            if self.selector.xys[self.selector.ind].size > 0:
                mod, ys = draw_interpolant(self.ui.figure.gca(),self.selector.xys[self.selector.ind],self.visible_data,self.ui.offset_sb.value())
            else:
                return
            self.ui.modulus_sb.setValue(mod)
            if ys is not None:
                self.ui.ys_sb.setValue(ys) #and should be a good point to tell the user that their Rp value hasn't returned anything
            self.ui.modulus_sb.setEnabled(True)
            self.ui.ys_sb.setEnabled(True)
            # selector.disconnect() #disabled to allow for reselecting points
            self.ui.canvas.draw()
        elif key == Qt.Key_F1:
            pass #help?
        elif key == Qt.Key_F5:
            self.update_plot()
        elif key == Qt.Key_A:
            if self.ui.cycle_slider.value()[0] < (self.ui.cycle_slider.value()[1]-0.25):
                self.ui.cycle_slider.setValue((self.ui.cycle_slider.value()[0], self.ui.cycle_slider.value()[1]-0.25))
        elif key == Qt.Key_W:
            if (self.ui.cycle_slider.value()[0]+0.25) < self.ui.cycle_slider.value()[1]: 
                self.ui.cycle_slider.setValue((self.ui.cycle_slider.value()[0]+0.25, self.ui.cycle_slider.value()[1]))
        elif key == Qt.Key_D:
            if self.ui.cycle_slider.value()[0] < (self.ui.cycle_slider.value()[1]+0.25):
                self.ui.cycle_slider.setValue((self.ui.cycle_slider.value()[0], self.ui.cycle_slider.value()[1]+0.25))
        elif key == Qt.Key_S:
            if (self.ui.cycle_slider.value()[0]-0.25) < self.ui.cycle_slider.value()[1]:
                self.ui.cycle_slider.setValue((self.ui.cycle_slider.value()[0]-0.25, self.ui.cycle_slider.value()[1]))

    def run(self):
        self.update_plot() #clear last fit
        #retrieve and sanitize bounds from respective widgets
        
        if not self.ui.ignore_bounds_cb.isChecked():
            bounds = {}
            for key in self.ui.iso_boundary_widget.val_dict.keys():
                try:
                    text = self.ui.iso_boundary_widget.val_dict[key].text().split(',')
                    bounds[key] = tuple([float(val) for val in text])
                except Exception as e:
                    print('Formatting error on boundaries.\n', e)
                    return
            for key in self.ui.kin_boundary_widget.val_dict.keys():
                try:
                    text = self.ui.kin_boundary_widget.val_dict[key].text().split(',')
                    bounds[key] = tuple([float(val) for val in text])
                except Exception as e:
                    print('Formatting error on boundaries.\n', e)
                    return
        else:
            bounds = None
        
        param_dict = dict(self.ui.iso_input, **self.ui.kin_input)
        param_key = ['C1', 'r1', 'C2', 'r2', 'C3', 'r3', 'Q', 'b']
        params = []
        for key in param_key:
            params.append(param_dict[key].value())
        params = np.asarray(params)
        
        self.cycle_value_change() #in case things have changed
        ind = np.nonzero((self.cycles<=self.ui.cycle_slider.value()[1]))
        ind = [int(x) for x in ind[0]]
        data = self.alldata[0:self.cycle_ind[ind[-1]][1],1::]
        data_ind = reduce_data(data,self.ui.reduce_data_sb.value())
        
        
        self.thread = execute_LM_bounded(
        self.ui.iter_sb.value(),
        params,
        data[data_ind,0]/100,
        data[data_ind,1],
        self.ui.modulus_sb.value()*1000,
        self.ui.ys_sb.value(),
        bounds,
        self.ui.damping_sb.value()*1e-6)
        
        self.thread._signal.connect(self.signal_accept)
        self.thread.start()
        self.ui.pbar.setTextVisible(True)
        self.ui.pbar.setStyleSheet("")
        self.ui.pbar.setRange(0,0)
        
    def signal_accept(self,msg):
        
        if int(msg) == 100:
            self.ui.pbar.setRange(0,100)
            self.ui.pbar.setValue(0)
            self.ui.pbar.setFormat("Complete")
            self.ui.pbar.setStyleSheet("QProgressBar"
              "{"
              "background-color: lightgreen;"
              "border : 1px"
              "}")
        #update ui
        param_key = ['C1', 'r1', 'C2', 'r2', 'C3', 'r3', 'Q', 'b']
        new_params = dict(zip(param_key, self.thread.param_result))
        for key in new_params.keys():
            print(f"{key}: {new_params[key]:.4f}")
        #update ui with new values
        for key in self.ui.iso_input.keys():
            if key in new_params:
                self.ui.iso_input[key].setValue(new_params[key])
        for key in self.ui.kin_input.keys():
            if key in new_params:
                self.ui.kin_input[key].setValue(new_params[key])

        self.draw_chaboche()

class execute_LM_bounded(QThread):
    '''
    Runs the LM_bounded method in a separate thread
    '''
    _signal = pyqtSignal(int)
    def __init__(self, _iter, params, strain, stress, mod, ys, _lambda, bounds):
        super(execute_LM_bounded, self).__init__()
        self._iter = _iter
        self.params = params
        self.strain = strain
        self.stress = stress
        self.mod = mod
        self.ys = ys
        self._lambda = _lambda
        self.bounds = bounds
        
    def run(self):
        self.param_result = LM_bounded(
        self._iter,
        self.params,
        self.strain,
        self.stress,
        self.mod,
        self.ys,
        self._lambda,
        self.bounds)

        self._signal.emit(100)

def make_boundary_button(parent, content='isotropic'):

    pixmapi = getattr(QtWidgets.QStyle, 'SP_FileDialogDetailedView')
    icon = parent.style().standardIcon(pixmapi)
    drop_button = QtWidgets.QToolButton()
    drop_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
    # drop_button.setCheckable(True)
    drop_button.setMenu(QtWidgets.QMenu(drop_button))
    action = QtWidgets.QWidgetAction(drop_button)
    if content == "isotropic":
        parent.iso_boundary_widget = boundary_value_box(parent,content)
        action.setDefaultWidget(parent.iso_boundary_widget)
    else:
        parent.kin_boundary_widget = boundary_value_box(parent,content)
        action.setDefaultWidget(parent.kin_boundary_widget)
    drop_button.menu().addAction(action)
    drop_button.setIcon(icon)
    drop_button.setToolTip('Set boundaries for fitting: with the format of min,max')
    return drop_button

class boundary_value_box(QtWidgets.QWidget):
    def __init__(self, parent, cond, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        
        self.val_dict = {}
        layout = QtWidgets.QGridLayout()
        if cond == 'isotropic':
            iso_labels = ['Q','b']
            default_vals = [(10,150),(1,10)]
            i = 0
            for entry in iso_labels:
                layout.addWidget(QtWidgets.QLabel(entry),i,0,1,1)
                self.val_dict[entry] = QtWidgets.QLineEdit()
                self.val_dict[entry].setText(str(default_vals[i]).strip('()'))
                layout.addWidget(self.val_dict[entry],i,1,1,1)
                i+=1
        
        else:
            kin_labels = ['C1', 'C2', 'C3', 'C4', 'r1', 'r2', 'r3', 'r4']
            kin_text = ['C\u2081','C\u2082', 'C\u2083', 'C\u2084', 'r\u2081','r\u2082', 'r\u2083', 'r\u2084']
            # default_vals = [(10000, 100000), (1000, 50000), (100, 10000), (0,0), (100, 2000), (10, 500), (1, 100), (0, 0)]
            default_vals = [(10000, 100000), (1000, 50000),(100, 10000), (0, 0), (100, 2000), (10, 500), (1, 100), (0, 0)]
            i = 0
            for entry in kin_labels:
                self.val_dict[entry] = QtWidgets.QLineEdit()
                self.val_dict[entry].setText(str(default_vals[i]).strip('()'))
                if i <= 3:
                    layout.addWidget(QtWidgets.QLabel(kin_text[i]),i,0,1,1)
                    layout.addWidget(self.val_dict[entry],i,1,1,1)
                else:
                    layout.addWidget(QtWidgets.QLabel(kin_text[i]),i-4,2,1,1)
                    layout.addWidget(self.val_dict[entry],i-4,3,1,1)
                i+=1
        self.setLayout(layout)

#code block for abaqus input deck
# *PLASTIC, HARDENING=COMBINED, DATA TYPE=PARAMETERS, NUMBER BACKSTRESSES=2
# *****************************************
# ** non-linear kinematic hardening based 
# ** on <5% plastic strain data from protocol
# ** 1e-9 is effectively zero
# *****************************************
# ** Sy, C1, gamma1, C2, gamma2, T (Pa,°C)
# 216.5e6, 156435e6, 1410.85, 6134e6, 47.19, 20  
# 165.6e6, 100631e6, 1410.85, 5568e6, 47.19, 275 
# 147.7e6, 64341e6,  1410.85, 5227e6, 47.19, 550 
# 117.3e6, 56232e6,  1410.85, 4108e6, 47.19, 750 
# 114.1e6, 49588e6,  1410.85, 292.1e6,  47.19, 900 
# 54.9e6,  1e-9,      1410.85, 1e-9,    47.19, 1000
# 34.0e6,  1e-9,      1410.85, 1e-9,    47.19, 1100
# 3.7e6,   1e-9,      1410.85, 1e-9,    47.19, 1400
# *CYCLIC HARDENING, PARAMETERS
# **Q_inf, b (Pa,°C)
# 216.5e6, 62.5e6, 6.9, 20  
# 165.6e6, 86.7e6, 6.9, 275 
# 147.7e6, 93.8e6, 6.9, 550 
# 117.3e6, 12.0e6, 6.9, 750 
# 114.1e6, 1e-9,  6.9,   900 
# 54.9e6,  1e-9,  6.9,   1000
# 34.0e6,  1e-9,  6.9,   1100
# 3.7e6,   1e-9,  6.9,   1400


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    window = standalone_app()
    window.show()
    ret = app.exec()
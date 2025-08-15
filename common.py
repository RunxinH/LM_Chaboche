import os
import numpy as np

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector, RectangleSelector

from PyQt6 import QtWidgets

def get_file(*args):
    '''
    Returns absolute path to filename and the directory it is located in from a PyQt5 filedialog. First value is file extension, second is a string which overwrites the window message.
    '''
    ext = args[0]
    if len(args)>1:
        launchdir = args[1]
    else: launchdir = os.getcwd()
    ftypeName={}
    ftypeName['*.csv']=["Select WaveMatrix record:", "*.csv", "Comma separated value file"]
    
    filer = QtWidgets.QFileDialog.getOpenFileName(None, ftypeName[ext][0], 
         launchdir,(ftypeName[ext][2]+' ('+ftypeName[ext][1]+');;All Files (*.*)'))
    
    if filer[0] == '':
        return None
    else:
        return filer[0]

class select_from_collection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, sel, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()

        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))
        
        if sel == "lasso":
            self.lasso = LassoSelector(ax, onselect=self.onselect)
        else:
            self.lasso = RectangleSelector(ax, self.rect_onselect)
        self.ind = []

    def rect_onselect(self,eclick,erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        path = Path(np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]]))
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        
    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def line_intersection(line1, line2):
    '''
    Returns the x,y intersection intersection of line1 and line2 if it exists.
    Params:
    line1, line2: tuple pairs of x,y points
    Returns:
    x & y values of intersection
    '''

    x1, x2, x3, x4 = line1[0][0], line1[1][0], line2[0][0], line2[1][0]
    y1, y2, y3, y4 = line1[0][1], line1[1][1], line2[0][1], line2[1][1]

    dx1 = x2 - x1
    dx2 = x4 - x3
    dy1 = y2 - y1
    dy2 = y4 - y3
    dx3 = x1 - x3
    dy3 = y1 - y3

    det = dx1 * dy2 - dx2 * dy1
    det1 = dx1 * dy3 - dx3 * dy1
    det2 = dx2 * dy3 - dx3 * dy2

    if det == 0.0:  # lines are parallel
    
        if np.array_equal(line1[0], line2[0]) or np.array_equal(line1[1], line2[0]):
            return line2[0]
        elif np.array_equal(line1[0], line2[1]) or np.array_equal(line1[1], line2[1]):
            return line2[1]

        return None #no intersection

    s = det1 / det
    t = det2 / det

    if 0.0 < s < 1.0 and 0.0 < t < 1.0:
        return x1 + t * dx1, y1 + t * dy1


def split_cycles(data,quartering = True):
    #assumes incoming np array data has column 0 being cycle, 1 is strain, 2 is stress
    part_cycles = np.unique(data[:,0])
    
    if not quartering:
        x = np.mod(part_cycles, 1)
        mask = (x == 0)
        part_cycles = part_cycles[mask]

    #separate record into a list of strain/stress
    # ss = []
    cycle_ind = []
    for i in range(len(part_cycles[1:-1])):
        #(a[:,0] >= i) and (a[:,0] < i+1)
        mask = (data[:,0] >= part_cycles[i]) & (data[:,0] < part_cycles[i+1])
        local_ind = np.nonzero(mask)[0]
        cycle_ind.append([int(local_ind[0]), int(local_ind[-1])])
        # ss.append(np.column_stack((data[mask,1], data[mask,2])))
    #ending at
    # (a[:,0] >= n)
    mask = (data[:,0] >= part_cycles[-1])
    local_ind = np.nonzero(mask)[0]
    cycle_ind.append([int(local_ind[0]), int(local_ind[-1])])
    # print(cycle_ind)
    # ss.append(np.column_stack((data[mask,1], data[mask,2])))
    return cycle_ind, part_cycles[1::] 


def reset_alpha(ax, collection):
    fc = collection.get_facecolors()
    npts = len(collection.get_offsets())
    if len(fc) == 0:
        raise ValueError('Collection must have a facecolor')
    elif len(fc) == 1:
        fc = np.tile(self.fc, (npts, 1))
    
    fc[:, -1] = 1
    collection.set_facecolors(fc)
    canvas = ax.figure.canvas
    canvas.draw_idle()
    
def draw_interpolant(ax, pts, data, offset):
    '''
    Accepts a matplotlib axis, some selected points from stress/strain data (MPa vs. %) and a strain offset in microstrain
    '''
    
    x = pts[:,0]
    y = pts[:,1]
    coeff = np.polyfit(x,y,1)
    compression = False
    #determine if T or C based on stresses increasing or decreasing in data
    if (pts[int(len(pts)*0.75),1] - pts[int(len(pts)*0.25),1]) < 0:
        #it's compression being targeted
        offset = -offset/10000
        y_fit = np.linspace(max(data[:,1]), min(data[:,1]),10)
        x_fit = (y_fit -coeff[1])/coeff[0]
        compression = True
    else:
        offset = offset/10000
        x_fit = np.linspace(min(data[:,0]),(max(data[:,1])-coeff[1])/coeff[0],10)
        y_fit = coeff[0]*x_fit + coeff[1]
        # print('Tension')
        
    #find intersection of offset interpolant with each line segment in data
    i = 0
    intersect = [None, None]
    for i in range(len(data)-1):
        test = [[data[i,0],data[i,1]], [data[i+1,0],data[i+1,1]]]
        intersect_res = line_intersection([[x_fit[0]+offset,y_fit[0]], [x_fit[-1]+offset,y_fit[-1]]], test)
        if intersect_res is not None:
            intersect = intersect_res
    
    
    interp = ax.plot(x_fit+offset,y_fit,'r--', markerfacecolor = 'None',
    label = 'm = %0.4f, b=%0.4f'%(coeff[0],coeff[1]))
    try:
        interp_pt = ax.plot(intersect[0], intersect[1], 'ro')
        canvas = ax.figure.canvas
        canvas.draw_idle()
    except Exception as e:
        print('yield point not found;\n %s'%e)
    
    if compression:
        return (coeff[0]*0.1), max(data[:,1])-intersect[1]
    else:
        return (coeff[0]*0.1), intersect[1]


def reduce_data(pnts, val):
    '''
    returns an index of reduced rows on an Nx2 np array based on val: a percentage of points to retain
    '''
    localind = np.arange(0, len(pnts), 1, dtype=int)
    
    #re-index
    red = float(val)/100
    ind = localind[np.arange(0,len(localind),int(1/red),dtype=int)]
    
    return ind
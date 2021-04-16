"""
GUI for analysis AS-OCT scans

Before running the script, it is important that:
- The libraries from the requirements.txt file are installed

Author: R. Lucassen (r.t.lucassen@student.tue.nl)
"""

import os
# turn of all warnings and information messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from scipy.ndimage import zoom
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.colors import ListedColormap
from PIL import Image, ImageTk

# configure root and start the GUI by creating an instance of the GUI class
def start_GUI():
    root = tk.Tk()
    GUI(root)

class GUI:
    # class variables for image processing
    N = 16                                  # number of B-scans
    network_path = 'network/network.h5'     # path to stored network
    image_size = (384, 960)                 # size of centered B-scan before left and right crop
    crop_left = (18, 274, 0, 512)           # left crop information (top row, bottom row, left most column, right most column) 
    crop_right = (18, 274, 448, 960)        # left crop information (top row, bottom row, left most column, right most column)
    batch = 1                               # size of batch for predicting delineation
    split = 362                             # position where to split the network prediction to obtain the input and output delineation
    inside_x = np.arange(150,810)           # x-coordinates of inside delineation
    outside_x = np.arange(80,880)           # x-coordinates of outside delineation
    kernel_size=71                          # kernel size for averaging filter to smooth coefficient of proportionality
    padding_mean=15                         # amount of padding for array with coeffient of proportionality
    lim = (100,700)                         # x-coordinates at the outsides of the 9 mm diameter region (max range is 0-800)
    fact = 15                               # factor to multiply pixel value with to find the values in micrometer (1px = 15 micrometer)
    X = 10                                  # factor for interpolation between B-scans in pachymetry map
    pachy_map = [200, 1400, 50, 13, 1150]   # pachymetry map settings (lowest thickness value, highest thickness value, interval for colors in colormap, number of color bar ticks, averages are displayed in white above this value)
    diff_map = [-300, 300, 25, 13, 150]     # differential pachymetry map settings (lowest thickness value, highest thickness value, interval for colors in colormap, number of color bar ticks, averages are displayed in white above or below this value)
    page_text=["Add AS-OCT B-scans", "Add AS-OCT B-scans", "Differential pachymetry map"] # text for page buttons at the top

    def __init__(self, root):
        """ Configure attributes and start the setup of the Tkinter window.
        """  
        # setup GPU
        self.gpu_setup()

        # create an empty dictionary to add in loaded information after processing of the AS-OCT scan
        self.data = [None, None]
        # store a reference and the placement information for all canvases/widgets in the dictionary
        self.dictionary = dict()
        # use lists to keep track of placed widgets and canvases
        self.placed_canvas = []
        self.placed_widget = []
        self.diff_pachy_button = False

        # class variables for keeping track of states
        self.idx = 0                        # track the B-scan to display
        self.visible = True                 # track whether all plotted lines in the B-scan must be visible or not
        self.total_lines = 0                # track the total number of files 
        self.dashed_lines = tk.IntVar()     # track whether the 9 mm dashed lines must be displayed
        self.thickness_lines = tk.IntVar()  # track whethet the thickness lines must be displayed
        self.space = 10                     # one in every 'mod' x-coordinates, the thickness line will be displayed
        self.index = -1                     # track page             

        # create root    
        self.root = root
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg='white')
        self.config()

        # start root after setup
        self.root.mainloop()

    def gpu_setup(self):
        """ Setup for GPU (code from: https://www.tensorflow.org/guide/gpu).
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    
    def close_gui(self):
        """ Function to properly close tkinter window.
        """
        self.root.quit()
        self.root.destroy()
    
    def config(self):
        """ Place page buttons in window to load in new patient information.
        """
        # create button and add place it on the window
        close_button = tk.Button(master=self.root, text="Quit", bg='white', font=("Open Sans", 12), command=self.close_gui)
        close_button.place(relx=0.96, rely=0.015, relwidth=0.03, relheight=0.03)      
        
        # create button and add place it on the window. Also create a text variable to be able to change the text on the button.
        text_0 = tk.StringVar()
        text_0.set(self.page_text[0])
        page_0 = tk.Button(master=self.root, textvariable=text_0, bg='white', font=("Open Sans", 12), command= lambda: self.button_click(0))
        page_0.place(relx=0.01, rely=0.015, relwidth=0.2, relheight=0.03)

        # create button and add place it on the window. Also create a text variable to be able to change the text on the button.
        text_1 = tk.StringVar()
        text_1.set(self.page_text[1])
        page_1 = tk.Button(master=self.root, textvariable=text_1, bg='white', font=("Open Sans", 12), command= lambda: self.button_click(1))
        page_1.place(relx=0.22, rely=0.015, relwidth=0.2, relheight=0.03)

        # create button but this one is not placed yet (compared to the other two). Also create a text variable to be able to change the text on the button.
        text_2 = tk.StringVar()
        text_2.set(self.page_text[2])
        page_2 = tk.Button(master=self.root, textvariable=text_2, bg='white', font=("Open Sans", 12), command= lambda: self.button_click(2))

        # the buttons and texts are stored in a list to easily access them
        self.text = [text_0, text_1, text_2]
        self.page = [page_0, page_1, page_2]


    def button_click(self, index):
        """ Responsible for the correct action after one of the buttons is pressed.
        """
        # create a variable with the old index and store the new index in self.index for reference
        old_index = self.index
        self.index = index

        # if no data is currently loaded in (which means that no figures are displayed), load in the data, configure the setup and place the figures/widgets
        if old_index == -1:
            # configure the button such that it is displayed as pressed
            self.page[self.index].config(bg='light grey', relief=tk.SUNKEN)
            loaded = self.load_data()
            # if the data was loaded in correctly, continue with the setup
            if loaded == True:
                self.pachymetry_map(self.index)
                self.setup()
                self.main_placement()
                self.pachymetry_map_changes(self.index, False)
                self.dictionary[f'Pachymetry_map{self.index}'][0].draw_idle()
            # if the data was not correctly loaded (e.g. selection window was cancelled), reset the button and index
            else:
                self.page[self.index].config(bg='white', relief=tk.RAISED)
                self.index = old_index
        # switch between AS-OCT scans
        elif (old_index == 0 or old_index == 1) and (self.index == 0 or self.index == 1):
            # if the data of the new scan was not yet loaded, load it first
            if self.data[self.index] == None:
                # display the button that was pressed as pressed and raise the previously pressed button
                self.page[old_index].config(bg='white', relief=tk.RAISED)
                self.page[self.index].config(bg='light grey', relief=tk.SUNKEN)
                # load the data
                loaded = self.load_data()
                # if the data was loaded in correctly, continue with the setup
                if loaded == True:
                    # create pachymetry map and display the current B-scan with the red line
                    self.pachymetry_map(self.index)
                    self.pachymetry_map_changes(self.index, False)
                    self.dictionary[f'Pachymetry_map{self.index}'][0].draw_idle()
                    # connect pachymetry map canvas to events
                    self.dictionary[f'Pachymetry_map{self.index}'][0].mpl_connect("scroll_event", self.onscroll)
                    self.dictionary[f'Pachymetry_map{self.index}'][0].mpl_connect('button_press_event', self.onclick)
                    # update the window
                    self.update_different_scan()
                    # create the differential pachymetry map
                    self.differential_pachymetry_map()
                # if the data was not correctly loaded (e.g. selection window was cancelled), reset the button and index
                else:
                    self.page[self.index].config(bg='white', relief=tk.RAISED)
                    self.page[old_index].config(bg='light grey', relief=tk.SUNKEN)
                    self.index = old_index
            # otherwise directly replace the AS-OCT scan directly
            elif old_index != self.index:
                self.page[old_index].config(bg='white', relief=tk.RAISED)
                self.page[self.index].config(bg='light grey', relief=tk.SUNKEN)
                self.update_different_scan()
        # switch from differential pachymetry map page to one of the AS-OCT scan page        
        elif old_index == 2 and self.index != 2:
            # configure buttons
            self.page[old_index].config(bg='white', relief=tk.RAISED)
            self.page[self.index].config(bg='light grey', relief=tk.SUNKEN)
            # in the differential pachymetry map, no red lines for the current B-scan are displayed, which is why we correct for this with 'remove=False' below
            self.pachymetry_map_changes(0, remove=False)
            self.pachymetry_map_changes(1, remove=False)
            self.main_placement()
        # switch from one of the AS-OCT scan page to differential pachymetry map page 
        elif self.index == 2 and old_index != 2:
            # configure buttons
            self.page[old_index].config(bg='white', relief=tk.RAISED)
            self.page[self.index].config(bg='light grey', relief=tk.SUNKEN)
            # in the differential pachymetry map, no red lines for the current B-scan are displayed, which is why these are not plotted because of 'to_diff=True'
            self.pachymetry_map_changes(0, to_diff=True)
            self.pachymetry_map_changes(1, to_diff=True)
            self.diff_pachymetry_placement()
        else:
            print('Unknown case')

        # if both AS-OCT scans are loaded, place the differential pachymetry map button next to it
        if self.data[0] != None and self.data[1] != None:
            if self.data[0][0][:5] == self.data[1][0][:5] and self.diff_pachy_button == False:
                self.page[2].place(relx=0.43, rely=0.015, relwidth=0.2, relheight=0.03)
                self.diff_pachy_button = True

    def load_data(self):
        """ Load B-scans after button click.
        """
        # while loop for selection of files    
        correct_selection = False
        while correct_selection == False:
            # select all images, the directories are stored in a list
            try:
                dirs = filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select files",filetypes = [("images",".png")])
            # if the cancel button was pressed, no values are returned. This value error is catched with the exception
            except ValueError:
                dirs = None               
            # check if the images are correct
            correct_selection, scan_name = self.check_files(dirs)
            # if the file selection window was canceled, then correct_selection is equal to 'QUIT' and the while loop should be broken
            if correct_selection == 'QUIT':
                return False  
        
        # if a correct selection of images was made, continue with the setup       
        if correct_selection == True:
            # create dataframe to store information
            images, data, pachymetry_data = self.prepare_data(dirs)
            self.text[self.index].set(scan_name)
            self.data[self.index] = (scan_name, images, data, pachymetry_data)

            return True

    
    def check_files(self, dirs):
        """ Check if all files are from the same patient and have the correct numbers.
        """
        # if the file selection window was closed and dirs is equal to None, return 'QUIT'
        if dirs == None:
            return 'QUIT', None
        # isolate the numbers of the filenames
        filenames = [file.split('/')[-1] for file in dirs]
        names = [file[:18] for file in filenames]
        numbers = [int(file[-7:-4]) for file in filenames]
        numbers.sort()

        # find the sizes of the images
        sizes = set()
        for file in dirs:
            img = Image.open(file)
            width, height = img.size
            sizes.add((height, width))

        # check if the selection was canceled
        if len(filenames) == 0:
            return 'QUIT', None
        # check if 16 files were selected
        elif len(filenames) != self.N:
            messagebox.showerror("Error", f"There were {len(filenames)} files selected. The program expects {self.N} files.")
            return False, None
        # check if the numbers of the filenames correspond to 0-15
        elif np.arange(self.N).tolist() != numbers:
            messagebox.showerror("Error", f"The file numbers did not consist of a unique range between 0 and {self.N-1}.")
            return False, None
        # check if all the names (without the numbers) are equal
        elif len(set(names)) > 1:
            messagebox.showerror("Error", f"The filenames did not correspond to the same patient.")
            return False, None
        # check if all images have the same size
        elif len(sizes) > 1:
            messagebox.showerror("Error", f"The image files did not all have the same height and/or width.")
            return False, None 
        # if everything is correct
        else:
            self.size = sizes.pop()
            return True, names[0][:-3]
    
    def prepare_data(self, dirs):
        """ Load the images, load the tensorflow model, predict the lines, calculate the distances and return the data.
        """
        # create dataframe
        df = pd.DataFrame(dirs, columns=['directories'])
        df['filenames'] = [file.split('/')[-1] for file in dirs]

        # allocate memory for images and crops
        images = np.zeros((self.N, self.size[0], self.size[1]))
        crops = np.zeros((self.N*2, self.crop_left[1]-self.crop_left[0], self.crop_left[3]-self.crop_left[2]))
        
        # load images and create crops
        for idx in np.arange(self.N):
            image = plt.imread(df['directories'][idx])
            images[idx,:,:] = image
            crops[idx,:,:] = image[self.crop_left[0]:self.crop_left[1], self.crop_left[2]:self.crop_left[3]]
            crops[idx+self.N,:,:] = np.flip(image[self.crop_right[0]:self.crop_right[1], self.crop_right[2]:self.crop_right[3]], axis=1)
        # expand the dimension for prediction
        crops = np.expand_dims(crops, axis=-1)

        # load tensorflow network and predict the values for the crops. Add the predictions to the empty list.
        network = tf.keras.models.load_model(self.network_path)
        inside_crop_predictions = []
        outside_crop_predictions = []
        # loop over the images in batches
        for i in np.arange(0,self.N*2,self.batch):
            # predict the inside and outside delineation
            prediction_batch = network.predict(crops[i:i+self.batch,:,:,:])
            # loop over the images in the batch
            for j in np.arange(self.batch):
                # process the prediction (different for left and right crop)
                prediction = prediction_batch[j]
                if i+j >= self.N:
                    inside_crop_predictions.append(np.flip(prediction[:self.split]))
                    outside_crop_predictions.append(np.flip(prediction[self.split:]))
                else:
                    inside_crop_predictions.append(prediction[:self.split])
                    outside_crop_predictions.append(prediction[self.split:])

        # combine the crops for the full images. Average over the overlapping parts. Add the predictions to the empty list.
        inside_predictions = []
        outside_predictions = []
        overlap = self.crop_left[3]-self.crop_right[2]
        for idx in np.arange(self.N):
            inside_overlap = np.mean((inside_crop_predictions[idx][-overlap:], inside_crop_predictions[idx+self.N][:overlap]), axis=0)
            inside_predictions.append(np.concatenate((inside_crop_predictions[idx][:-overlap], inside_overlap, inside_crop_predictions[idx+self.N][overlap:]), axis=0)+self.crop_left[0])
                    
            outside_overlap = np.mean((outside_crop_predictions[idx][-overlap:], outside_crop_predictions[idx+self.N][:overlap]), axis=0)
            outside_predictions.append(np.concatenate((outside_crop_predictions[idx][:-overlap], outside_overlap, outside_crop_predictions[idx+self.N][overlap:]), axis=0)+self.crop_left[0])

        # save the predicted interfaces in the dataframe
        df['inside_y_pred'] = inside_predictions
        df['outside_y_pred'] = outside_predictions

        # find the corneal thickness between the delineated interfaces. Add the predictions to the empty list.
        thicknesses = []
        all_thickness_lines = []
        for idx in np.arange(self.N):
            labels = (np.array(self.inside_x), df['inside_y_pred'][idx], np.array(self.outside_x), df['outside_y_pred'][idx])
            thickness, thickness_lines = self.measure_thickness(labels)
            thicknesses.append(thickness)
            all_thickness_lines.append(thickness_lines)
        
        # save the thickness and the lines in the dataframe
        df['thickness'] = thicknesses
        df['thickness_lines'] = all_thickness_lines   

        # define the range of x-coordinates used for the pachymetry map
        x_range = self.lim[1]-self.lim[0]

        # Create a grid of values for the radial distance r and the angle theta
        angle = np.radians(np.linspace(0, 360, 2*self.N+1))
        x = np.arange(0, x_range//2)
        r, theta = np.meshgrid(x, angle)

        filenames = df['filenames'].to_list()

        participant = filenames[0][:15]

        # Store the thickness values in an array
        values = np.empty((r.shape[0], x.shape[0]))
        for i in np.arange(self.N):
            num = '0'*(3-len(str(i)))+str(i)
            idx=filenames.index(f'{participant}_im{num}.png')
            array = df['thickness'][idx]
            correction = (array.shape[0]-x_range)//2
            array = array[correction:(array.shape[0]-correction)]
            values[i,:] = array[x_range//2:]
            values[i+self.N,:] = np.flip(array[:x_range//2])

        # add the first values to the last position as well and multiply with the conversion factor
        values[-1,:] = values[0,:]
        values *= self.fact

        # cubic interpolation between the thickness values
        values = zoom(values, (self.X,1))
        r = zoom(r, (self.X,1))
        theta = zoom(theta, (self.X,1))

        return images, df, (r, theta, values)  

    def measure_thickness(self, labels):
        """ Measure the thickness between the delineated interfaces.
        """
        # raise an error if the kernel size is even
        if self.kernel_size % 2 == 0:
            raise ValueError('kernel size must be an odd number')
        # retrieve annotation data
        inside_x = labels[0]
        inside_y = labels[1]
        outside_x = labels[2]
        outside_y = labels[3]
        # allocate memory for the distances
        dist = np.zeros(outside_x.shape)
        lines = [ [[None, None], [None, None]] for _ in np.arange(outside_x.shape[0])]
        # coefficient of proportionality of outside line
        dydx_outside = outside_y[1:]-outside_y[:-1]
        # add padding to the outside of the derivative array
        pad = int((self.kernel_size-1)/2)
        dydx_outside = np.concatenate(
            (np.ones(pad+1,)*np.mean(dydx_outside[:self.padding_mean]), dydx_outside, (np.ones(pad,)*np.mean(dydx_outside[-self.padding_mean:dydx_outside.shape[0]]))))
        # average coefficient of proportionality using sliding kernel approach
        dydx_corr_outside = np.correlate(
            dydx_outside, np.ones((self.kernel_size,))/self.kernel_size)+1e-8

        # loop over the indices of the inside x-coordinates
        for idx in np.arange(len(outside_x)):
            # find the inside x and y coordinates
            x0, y0, dydx = outside_x[idx], outside_y[idx], dydx_corr_outside[idx]
            # find the normal
            normal = (-1/dydx)*inside_x + (y0-(-1/dydx)*x0)
            # find out where the distance goes from positive to negative or vice versa to know where the lines intersect
            i = np.argwhere(np.diff(np.sign(inside_y - normal)))
            # Add none if there is no intersection. Store the distance if there is an intersection.
            if len(i) == 0:
                dist[x0-80] = None
            else:
                dist[x0-80] = np.sqrt((x0-inside_x[i][0])**2 + (y0-inside_y[i][0])**2)
                lines[x0-80] = [[x0, inside_x[i][0][0]], [y0, inside_y[i][0][0]]]

        return dist, lines

    def pachymetry_map(self, idx):
        """ Function to plot the thickness from radial B-scans in a pachymetry map.
        """
        # load the data for the pachymetry map and define the key to later store the pachymetry map variables in the dictionary
        r, theta, values = self.data[idx][3]
        key = f'Pachymetry_map{idx}'

        # create colormap (Modified version of Turbo: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html)
        cmap = 0.9*np.flip(np.array([[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]), axis=0)
        cmap = cmap[64:, :]

        # define the range of x-coordinates used for the pachymetry map
        x_range = self.lim[1]-self.lim[0]

        # create an empty list to store the averages
        averages = []
        # add the average of the apex to the list
        averages.append(np.mean(values[:, 0:67]))
        # define variables for calculating the averages of the regions
        start=0
        end=8*self.X+1
        # for the inner ring
        for _ in np.arange(4):
            averages.append(np.mean(values[start:end, 0:100]))
            start +=8*self.X
            end+=8*self.X
        # for the outer two rings
        for i in [100, 200]:
            start = 0
            end = 4*self.X+1
            for _ in np.arange(8):
                averages.append(np.mean(values[start:end, i:i+100]))
                start +=4*self.X
                end+=4*self.X
        
        # create the figure and configure settings
        self.fig2, self.ax2 = plt.subplots(subplot_kw=dict(projection='polar'))
        color_range = np.arange(self.pachy_map[0], self.pachy_map[1]+1, self.pachy_map[2])
        contour = self.ax2.contourf(theta, r, values, color_range, cmap=ListedColormap(cmap))
        plt.ylim([0,x_range//2])
        clb = plt.colorbar(contour, ticks=np.linspace(self.pachy_map[0], self.pachy_map[1], self.pachy_map[3]), pad=0.15)
        clb.ax.set_title('μm')
        self.ax2.grid(False)
        # define variables used for plotting the axes lines       
        r_axes=8
        theta_axes=3
        # plot the radial lines
        for i in np.arange(r_axes):
            plt.plot([i/r_axes*2*np.pi, i/r_axes*2*np.pi],[x_range//(2*theta_axes), x_range//2], linewidth=0.6, color='k')
        # plot the circular lines
        for j in np.arange(theta_axes+1):
            if j == np.arange(theta_axes)[-1]+1:
                plt.plot(np.radians(np.linspace(0, 360, 1000)), j*np.ones(1000)*x_range//(2*theta_axes), linewidth=2, color='k')
            else:
                plt.plot(np.radians(np.linspace(0, 360, 1000)), j*np.ones(1000)*x_range//(2*theta_axes), linewidth=0.6, color='k')
        # remove the axis ticks
        self.ax2.set_yticks([])
    
        # define variables for the positions of the average values in the pachymetry map
        y = [0, 60, 60, 60, 60, 150, 150, 150, 150, 150, 150, 150, 150, 250, 250, 250, 250, 250, 250, 250, 250]
        x = [0, (1/4)*np.pi, (3/4)*np.pi, (5/4)*np.pi, (7/4)*np.pi, (1/8)*np.pi, (3/8)*np.pi, (5/8)*np.pi, (7/8)*np.pi, (9/8)*np.pi, (11/8)*np.pi, (13/8)*np.pi, (15/8)*np.pi, (1/8)*np.pi, (3/8)*np.pi, (5/8)*np.pi, (7/8)*np.pi, (9/8)*np.pi, (11/8)*np.pi, (13/8)*np.pi, (15/8)*np.pi]
        # plot the average values
        for i in np.arange(len(x)):
            section_mean = int(round(averages[i]))
            # if the average is larger than a threshold value, display it in white instead of black
            if section_mean >= self.pachy_map[4]:
                self.ax2.text(x[i], y[i], section_mean, horizontalalignment='center', verticalalignment='center', color='white')
            else:
                self.ax2.text(x[i], y[i], section_mean, horizontalalignment='center', verticalalignment='center', color='black')

        # create canvas to place the pachymetry map at a later stage
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.root)
        self.dictionary[key] = (self.canvas2, [0.61, 0.05, 0.4, 0.48], [0, 0.05+(0.47*idx), 0.4, 0.48], self.ax2)

    def pachymetry_map_changes(self, idx, remove = True, to_diff=False):
        """ Function to plot the thickness from radial B-scans in a pachymetry map.
        """
        # get the axis to apply changes and load the data of the pachymetry map
        ax = self.dictionary[f'Pachymetry_map{idx}'][3]
        values = self.data[idx][3][2]

        # remove all text
        ax.texts = []
        # remove last two plotted lines
        if remove == True:
            ax.lines = ax.lines[:-2]

        # define the range of x-coordinates used for the pachymetry map
        x_range = self.lim[1]-self.lim[0]

        # create an empty list to store the averages
        averages = []
        # add the average of the apex to the list
        averages.append(np.mean(values[:, 0:67]))
        # define variables for calculating the averages of the regions
        start=0
        end=8*self.X+1
        # for the inner ring
        for _ in np.arange(4):
            averages.append(np.mean(values[start:end, 0:100]))
            start +=8*self.X
            end+=8*self.X
        # for the outer two rings
        for i in [100, 200]:
            start = 0
            end = 4*self.X+1
            for _ in np.arange(8):
                averages.append(np.mean(values[start:end, i:i+100]))
                start +=4*self.X
                end+=4*self.X
        
        # define variables used for plotting the axes lines       
        cross_sections=16
        theta_axes=3
        
        # plot the radial lines
        if to_diff == False:
            ax.plot([self.idx/cross_sections*np.pi, self.idx/cross_sections*np.pi],[x_range//(2*theta_axes), x_range//2], linewidth=2, color='r')
            ax.plot([self.idx/cross_sections*np.pi+np.pi, self.idx/cross_sections*np.pi+np.pi],[x_range//(2*theta_axes), x_range//2], linewidth=2, color='r')
        
        # dictionary with information what averages to leave out for a certain B-scan
        avg_dict = {0: [], 1: [], 2: [5,9,13,17], 3: [], 4: [], 5: [], 6: [6,10,14,18], 7: [], 8: [], 9: [], 10: [7,11,15,19], 11: [], 12: [], 13: [], 14: [8,12,16,20], 15: []}

        # define variables for the positions of the average values in the pachymetry map
        y = [0, 60, 60, 60, 60, 150, 150, 150, 150, 150, 150, 150, 150, 250, 250, 250, 250, 250, 250, 250, 250]
        x = [0, (1/4)*np.pi, (3/4)*np.pi, (5/4)*np.pi, (7/4)*np.pi, (1/8)*np.pi, (3/8)*np.pi, (5/8)*np.pi, (7/8)*np.pi, (9/8)*np.pi, (11/8)*np.pi, (13/8)*np.pi, (15/8)*np.pi, (1/8)*np.pi, (3/8)*np.pi, (5/8)*np.pi, (7/8)*np.pi, (9/8)*np.pi, (11/8)*np.pi, (13/8)*np.pi, (15/8)*np.pi]
        # plot the average values
        for i in np.arange(len(x)):
            section_mean = int(round(averages[i]))
            # if the selected page is the differential pachymetry map page (to_diff=True), then do not remove averages
            if to_diff == False:
                leave_out = avg_dict[self.idx]
            else:
                leave_out = []
            # plot all the average values
            if i not in leave_out:
                # if the average is larger than a threshold value, display it in white instead of black
                if section_mean >= self.pachy_map[4]:
                    ax.text(x[i], y[i], section_mean, horizontalalignment='center', verticalalignment='center', color='white')
                else:
                    ax.text(x[i], y[i], section_mean, horizontalalignment='center', verticalalignment='center', color='black')

    def differential_pachymetry_map(self):
        """ Function to plot the thickness from radial B-scans in a pachymetry map.
        """
        # load in the values from the other two pachymetry maps and take the difference
        r, theta, values1 = self.data[0][3]
        _, _, values2 = self.data[1][3]

        diff = values2-values1

        # create colormap (Modified version of Turbo: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html)
        cmap = np.array([[0, 18,     97],    [1, 20,     98],    [1, 21,     99],    [1, 23,    100],    [1, 24,    101],    [1, 26,    102],    [2, 28,    103],    [2, 29,    104],    [2, 31,    105],    [2, 32,    106],    [2, 34,    107],    [2, 35,    108],    [2, 37,    109],    [2, 39,    110],    [2, 40,    111],    [2, 42,    112],    [2, 43,    113],    [2, 45,    114],    [2, 46,    115],    [2, 48,    116],    [2, 49,    117],    [2, 51,    118],    [2, 52,    119],    [2, 54,    120],    [2, 55,    121],    [2, 57,    122],    [2, 58,    123],    [3, 60,    124],    [3, 62,    125],    [3, 63,    126],    [3, 65,    127],    [3, 66,    128],    [3, 68,    129],    [3, 69,    130],    [3, 71,    131],    [3, 73,    132],    [3, 74,    133],    [4, 76,    134],    [4, 77,    135],    [4, 79,    136],    [5, 81,    137],    [5, 82,    138],    [6, 84,    139],    [6, 86,    140],    [7, 87,    141],    [8, 89,    143],    [9, 91,    144],    [11, 93,   145],    [12, 94,   146],    [14, 96,   147],    [16, 98,   148],    [17, 100,  150],    [19, 102,  151],    [21, 103,  152],    [23, 105,  153],    [25, 107,  154],    [28, 109,  156],    [30, 111,  157],    [32, 113,  158],    [35, 115,  160],    [37, 117,  161],    [40, 119,  162],    [43, 121,  164],    [45, 123,  165],    [48, 125,  166],    [51, 127,  168],    [54, 129,  169],    [57, 131,  171],    [60, 133,  172],    [63, 135,  173],    [66, 137,  175],    [69, 139,  176],    [72, 141,  178],    [75, 144,  179],    [78, 146,  180],    [81, 148,  182],    [84, 150,  183],    [87, 152,  185],    [90, 154,  186],    [93, 156,  187],    [97, 158,  189],    [100, 160, 190],    [103, 162, 192],    [106, 164, 193],    [109, 166, 194],    [113, 168, 196],    [116, 170, 197],    [119, 172, 198],    [122, 174, 200],    [125, 176, 201],    [128, 178, 202],    [132, 180, 204],    [135, 182, 205],    [138, 184, 206],    [141, 186, 208],    [144, 188, 209],    [148, 190, 210],    [151, 192, 212],    [154, 194, 213],    [157, 196, 214],    [160, 197, 216],    [163, 199, 217],    [167, 201, 218],    [170, 203, 220],    [173, 205, 221],    [176, 207, 222],    [179, 209, 223],    [182, 211, 225],    [186, 213, 226],    [189, 214, 227],    [192, 216, 228],    [195, 218, 229],    [198, 219, 230],    [201, 221, 231],    [204, 223, 232],    [207, 224, 232],    [210, 225, 233],    [213, 227, 233],    [216, 228, 233],    [219, 229, 233],    [222, 230, 233],    [224, 230, 233],    [226, 231, 232],    [229, 231, 232],    [231, 231, 231],    [232, 231, 229],    [234, 230, 228],    [235, 230, 226],    [236, 229, 224],    [237, 228, 222],    [238, 227, 220],    [238, 225, 218],    [238, 224, 216],    [238, 222, 213],    [238, 221, 211],    [238, 219, 208],    [238, 217, 205],    [237, 215, 203],    [237, 213, 200],    [236, 211, 197],    [236, 209, 195],    [235, 208, 192],    [234, 206, 189],    [233, 204, 186],    [233, 202, 184],    [232, 200, 181],    [231, 198, 178],    [230, 196, 176],    [229, 193, 173],    [228, 191, 170],    [228, 190, 168],    [227, 188, 165],    [226, 186, 162],    [225, 184, 160],    [224, 182, 157],    [223, 180, 154],    [223, 178, 152],    [222, 176, 149],    [221, 174, 147],    [220, 172, 144],    [219, 170, 141],    [219, 168, 139],    [218, 166, 136],    [217, 164, 134],    [216, 162, 131],    [215, 160, 129],    [214, 159, 126],    [214, 157, 124],    [213, 155, 121],    [212, 153, 119],    [211, 151, 116],    [211, 149, 114],    [210, 148, 112],    [209, 146, 109],    [208, 144, 107],    [207, 142, 104],    [207, 140, 102],    [206, 139, 100],    [205, 137,   7],    [204, 135,   5],    [204, 133,   3],    [203, 131,   0],    [202, 130,   8],    [201, 128,   6],    [201, 126,   3],    [200, 124,   1],    [199, 123,   9],    [198, 121,   6],    [198, 119,   4],    [197, 117,   2],    [196, 116,   9],    [195, 114,   7],    [194, 112,   5],    [194, 110,   3],    [193, 109,   0],    [192, 107,   8],    [191, 105,   6],    [190, 103,   4],    [190, 101,   1],    [189, 100,   9],    [188, 98,    7],    [187, 96,    5],    [186, 94,    2],    [184, 92,    0],    [183, 90,    8],    [182, 88,    6],    [181, 85,    3],    [179, 83,    1],    [178, 81,    9],    [176, 79,    7],    [175, 76,    4],    [173, 74,    2],    [171, 72,    0],    [169, 69,    8],    [167, 67,    6],    [165, 64,    5],    [163, 62,    3],    [161, 60,    1],    [159, 57,    0],    [156, 55,    9],    [154, 53,    8],    [152, 51,    7],    [150, 49,    7],    [148, 47,    6],    [145, 45,    6],    [143, 43,    6],    [141, 41,    6],    [139, 39,    6],    [137, 38,    6],    [135, 36,    6],    [133, 34,    6],    [131, 33,    6],    [129, 31,    6],    [127, 30,    6],    [126, 29,    6],    [124, 27,    6],    [122, 26,    6],    [120, 24,    6],    [118, 23,    6],    [116, 21,    6],    [115, 20,    6],    [113, 19,    7],    [111, 17,    7],    [109, 16,    7],    [108, 14,    7],    [106, 13,    7],    [104, 12,    7],    [103, 10,    7],    [101, 9,     7],    [99, 7,      7],    [98, 6,      7],    [96, 4,      8],    [94, 3,      8],    [93, 2,      8],    [91, 1,      8],    [89, 0,      8]])/255

        # define the range of x-coordinates used for the pachymetry map
        x_range = self.lim[1]-self.lim[0]

        # create an empty list to store the averages
        averages = []
        # add the average of the apex to the list
        averages.append(np.mean(diff[:, 0:67]))
        # define variables for calculating the averages of the regions
        start=0
        end=8*self.X+1
        # for the inner ring
        for _ in np.arange(4):
            averages.append(np.mean(diff[start:end, 0:100]))
            start +=8*self.X
            end+=8*self.X
        # for the outer two rings
        for i in [100, 200]:
            start = 0
            end = 4*self.X+1
            for _ in np.arange(8):
                averages.append(np.mean(diff[start:end, i:i+100]))
                start +=4*self.X
                end+=4*self.X
        
        # create the figure and configure settings
        self.fig4, self.ax4 = plt.subplots(subplot_kw=dict(projection='polar'))
        color_range = np.arange(self.diff_map[0], self.diff_map[1]+1, self.diff_map[2])
        contour = self.ax4.contourf(theta, r, diff, color_range, cmap=ListedColormap(cmap))
        plt.ylim([0,x_range//2])
        clb = plt.colorbar(contour, ticks=np.linspace(self.diff_map[0], self.diff_map[1], self.diff_map[3]), pad=0.15)
        clb.ax.set_title('μm')
        self.ax4.grid(False)
        # define variables used for plotting the axes lines       
        r_axes=8
        theta_axes=3
        # plot the radial lines
        for i in np.arange(r_axes):
            plt.plot([i/r_axes*2*np.pi, i/r_axes*2*np.pi],[x_range//(2*theta_axes), x_range//2], linewidth=0.6, color='k')
        # plot the circular lines
        for j in np.arange(theta_axes+1):
            if j == np.arange(theta_axes)[-1]+1:
                plt.plot(np.radians(np.linspace(0, 360, 1000)), j*np.ones(1000)*x_range//(2*theta_axes), linewidth=2, color='k')
            else:
                plt.plot(np.radians(np.linspace(0, 360, 1000)), j*np.ones(1000)*x_range//(2*theta_axes), linewidth=0.6, color='k')
        # remove the axis ticks
        self.ax4.set_yticks([])
    
        # define variables for the positions of the average values in the pachymetry map
        y = [0, 60, 60, 60, 60, 150, 150, 150, 150, 150, 150, 150, 150, 250, 250, 250, 250, 250, 250, 250, 250]
        x = [0, (1/4)*np.pi, (3/4)*np.pi, (5/4)*np.pi, (7/4)*np.pi, (1/8)*np.pi, (3/8)*np.pi, (5/8)*np.pi, (7/8)*np.pi, (9/8)*np.pi, (11/8)*np.pi, (13/8)*np.pi, (15/8)*np.pi, (1/8)*np.pi, (3/8)*np.pi, (5/8)*np.pi, (7/8)*np.pi, (9/8)*np.pi, (11/8)*np.pi, (13/8)*np.pi, (15/8)*np.pi]
        # plot the average values
        for i in np.arange(len(x)):
            section_mean = int(round(averages[i]))
            # if the average is larger than a threshold value, display it in white instead of black
            if abs(section_mean) >= self.diff_map[4]:
                self.ax4.text(x[i], y[i], section_mean, horizontalalignment='center', verticalalignment='center', color='white', fontsize=12)
            else:
                self.ax4.text(x[i], y[i], section_mean, horizontalalignment='center', verticalalignment='center', color='black', fontsize=12)

        # create canvas to place the differential pachymetry map at a later stage
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.root)
        self.dictionary['Differential_pachymetry_map'] = (self.canvas4, [0.4, 0.2, 0.6, 0.6])

    def change_pachy_levels(self):
        """ Change the number of levels in the colormap and pachymetry map
        """
        # retrieve the entry after the button click
        # correct it if the values was too small (<1) or too large (>200, this value can be much larger, but no differences can be observed anymore) 
        entry = min(max(int(self.entry_pachy.get()),1),200)
        self.pachy_map[2] = (self.pachy_map[1]-self.pachy_map[0])/entry
        # loop over the two pachymetry maps (if the data exists)
        for i in np.arange(2):
            if self.data[i] != None:
                # destroy the previous version of the canvas
                self.dictionary[f'Pachymetry_map{i}'][0].get_tk_widget().destroy()
                # create the new canvas with updated settings
                self.pachymetry_map(i)
                # if the pachymetry map is currently displayed, place them on in the window again and account for the page for the red line
                if f'Pachymetry_map{i}' in self.placed_canvas:
                    self.placed_canvas.remove(f'Pachymetry_map{i}')
                    if self.index <2:
                        self.pachymetry_map_changes(i, remove=False)
                        self.place_canvas(f'Pachymetry_map{i}', 1)  
                    else:
                        self.pachymetry_map_changes(i, remove=False, to_diff=True)
                        self.place_canvas(f'Pachymetry_map{i}', 2)
                else:
                    self.pachymetry_map_changes(i, remove=False)  

    def change_diff_levels(self):
        """ Change the number of levels in the colormap and differential pachymetry map
        """
        # retrieve the entry after the button click
        # correct it if the values was too small (<1) or too large (>200, this value can be much larger, but no differences can be observed anymore) 
        entry = min(max(int(self.entry_diff.get()),1),200)
        self.diff_map[2] = (self.diff_map[1]-self.diff_map[0])/entry
        # destroy the previous version of the canvas and place the new version in the window        
        self.dictionary['Differential_pachymetry_map'][0].get_tk_widget().destroy()
        self.differential_pachymetry_map()
        self.placed_canvas.remove('Differential_pachymetry_map')
        self.place_canvas('Differential_pachymetry_map',1)  
    
    def setup(self):
        """ Creates figures for GUI and places them in the window.
        1 = B-scans
        2 = (differential) pachymetry map
        3 = Thickness profile plot
        """
        # retrieve data
        images = self.data[self.index][1]
        df = self.data[self.index][2]

        # create the figure with delineated B-scan plot
        self.fig1 = plt.Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title(f"{self.idx}/{self.N-1}", loc='right', fontsize=14, fontname='Open sans')
        self.img = self.ax1.imshow(images[self.idx,:,:], cmap='Greys_r')
        self.ax1.axis('off')
        self.ax1.set_ylim(self.image_size[0], 0)
        self.ax1.plot(self.inside_x, df['inside_y_pred'][self.idx], color='r', linewidth = 2)
        self.ax1.plot(self.outside_x, df['outside_y_pred'][self.idx], color='r', linewidth = 2)
        # create canvas to place the delineated B-scan plot at a later stage
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.root)
        self.dictionary['B_scan'] = (self.canvas1, [-0.06, 0.045, 0.70, 0.5], [-0.12, 0.045, 1.2, 0.8])

        # create the figure with the thickness profile plot
        self.fig3 = plt.Figure()
        px_radius = (self.lim[1]-self.lim[0])//2
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.grid(axis='y')
        self.ax3.set_xlim([-px_radius,px_radius])
        self.ax3.set_ylim([0, self.pachy_map[1]])
        self.ax3.set_xlabel('Distance from center (mm)')
        self.ax3.set_ylabel('Corneal thickness (μm)')
        self.ax3.set_xticks(np.arange(-px_radius, px_radius+1, 100))
        self.ax3.set_xticklabels([str(i*15/1000) for i in np.arange(-px_radius, px_radius+1, 100)])
        self.ax3.plot(np.arange(-px_radius, px_radius), df['thickness'][self.idx][self.lim[0]:self.lim[1]]*self.fact, color='k', linewidth = 1.5)
        # create canvas to place the thickness profile plot at a later stage
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.root)
        self.dictionary['Thickness_profile'] = (self.canvas3, [0.076, 0.50, 0.435, 0.45])

        # Below, all simple widgets are configured (look at the dictionary key to know wat widget it is)
        self.options_label = tk.Label(master=self.root, text='Visualization settings', background='white', font=("Open Sans", 18, "bold"))
        self.dictionary['Label_options'] = (self.options_label, [0.6, 0.55, 0.2, 0.05])
        
        self.dashed_lines_check = tk.Checkbutton(master=self.root, text='Show 9 mm diameter', background='white', activebackground='white', font=("Open Sans", 14), variable=self.dashed_lines, command=self.update_same_scan)
        self.dictionary['Check_9mm'] = (self.dashed_lines_check, [0.59, 0.6, 0.2, 0.05], [0.031, 0.76, 0.2, 0.05])

        self.thickness_lines_check = tk.Checkbutton(master=self.root, text='Show thickness lines', background='white', activebackground='white', font=("Open Sans", 14), variable=self.thickness_lines, command=self.update_same_scan)
        self.dictionary['Check_thickness_lines'] = (self.thickness_lines_check, [0.587, 0.65, 0.2, 0.05], [0.227, 0.76, 0.2, 0.05])

        self.remove_data_button = tk.Button(master=self.root, text='Remove data', background='white', font=("Open Sans", 14), command=self.remove_data)
        self.dictionary['Button_remove_data'] = (self.remove_data_button, [0.62, 0.85, 0.15, 0.05])

        self.full_screen_button = tk.Button(master=self.root, text='Full screen', background='white', font=("Open Sans", 12), command=self.full_screen)
        self.dictionary['Button_full_screen'] = (self.full_screen_button, [0.50, 0.50, 0.07, 0.03])

        self.main_screen_button = tk.Button(master=self.root, text='Main screen', background='white', font=("Open Sans", 12), command=self.main_placement)
        self.dictionary['Button_main_screen'] = (self.main_screen_button, [0.86, 0.77, 0.07, 0.03])

        self.entry_pachy = tk.Entry(master=self.root, borderwidth=1, background='white', font=("Open Sans", 14))
        self.entry_pachy.insert(0, str((self.pachy_map[1]-self.pachy_map[0])//self.pachy_map[2]))
        self.dictionary['Entry_pachy'] = (self.entry_pachy, [0.79, 0.71, 0.04, 0.03], [0.79, 0.83, 0.04, 0.03])

        self.entry_pachy_button = tk.Button(master=self.root, text='Apply entry', background='white', font=("Open Sans", 12), command=self.change_pachy_levels)
        self.dictionary['Button_entry_pachy'] = (self.entry_pachy_button, [0.84, 0.71, 0.07, 0.03], [0.84, 0.83, 0.07, 0.03])
        
        self.pachy_level_label = tk.Label(master=self.root, text='Pachymetry color levels:', background='white', font=("Open Sans", 14))
        self.dictionary['Label_pachy_levels'] = (self.pachy_level_label, [0.63, 0.7, 0.15, 0.05], [0.54, 0.82, 0.15, 0.05])

        self.entry_diff = tk.Entry(master=self.root, borderwidth=1, background='white', font=("Open Sans", 14))
        self.entry_diff.insert(0, str((self.diff_map[1]-self.diff_map[0])//self.diff_map[2]))
        self.dictionary['Entry_diff'] = (self.entry_diff, [0.79, 0.76, 0.04, 0.03], [0.79, 0.88, 0.04, 0.03])

        self.entry_diff_button = tk.Button(master=self.root, text='Apply entry', background='white', font=("Open Sans", 12), command=self.change_diff_levels)
        self.dictionary['Button_entry_diff'] = (self.entry_diff_button, [0.84, 0.76, 0.07, 0.03], [0.84, 0.88, 0.07, 0.03])

        self.diff_level_label = tk.Label(master=self.root, text='Differential pachymetry color levels:', background='white', font=("Open Sans", 14))
        self.dictionary['Label_diff_levels'] = (self.diff_level_label, [0.63, 0.75, 0.15, 0.05], [0.524, 0.87, 0.25, 0.05])

        self.map_label = tk.Label(master=self.root, text='Differential pachymetry map', background='white', font=("Open Sans", 14))
        self.dictionary['Label_map'] = (self.map_label, [0.565, 0.16, 0.2, 0.05])

    def place_canvas(self, obj, placement):
        """ Place canvas in the window based on placement information from the dictionary item of the canvas.
        """
        # retrieve the canvas from the dictionary
        var = self.dictionary[obj]
        # place the canvas in the window
        var[0].get_tk_widget().place(relx=var[placement][0], rely=var[placement][1], relwidth=var[placement][2], relheight=var[placement][3])
        # draw / initialize the canvas
        var[0].draw()
        # add the canvas to the list of placed canvases
        self.placed_canvas.append(obj)

    def place_widget(self, obj, placement):
        """ Place widget in the window based on placement information from the dictionary item of the widget.
        """
        # retrieve the widget from the dictionary
        var = self.dictionary[obj]
        # place the widget in the window
        var[0].place(relx=var[placement][0], rely=var[placement][1], relwidth=var[placement][2], relheight=var[placement][3])
        # add the widget to the list of placed canvases
        self.placed_widget.append(obj)

    def clear_window(self):
        """ Removes all placed canvases and widgets using the references that were made when placing the widgets.
        """
        # for all widgets in the list with placed widgets, forget the placement and empty the list
        if len(self.placed_widget) > 0:
            for name in self.placed_widget:
                widget = self.dictionary[name][0]
                widget.place_forget()
            
            self.placed_widget = []
        
        # for all canvases in the list with placed widgets, forget the placement and empty the list
        if len(self.placed_canvas) > 0:
            for name in self.placed_canvas:
                widget = self.dictionary[name][0]
                widget.get_tk_widget().place_forget()
            
            self.placed_canvas = []

    def full_screen(self):
        """ Clear the window and place all the canvases and widgets for the full screen delineated B-scan plot.
        """
        self.clear_window()
        self.place_canvas('B_scan',2)
        self.place_widget('Button_main_screen', 1)
        self.place_widget('Check_9mm', 2)
        self.place_widget('Check_thickness_lines', 2)

    def remove_data(self):
        """ Delete stored data and do changes to variables / states.
        """
        # remove data corresponding to one of the pages
        self.data[self.index] = None
        # reset the text of the page button
        self.text[self.index].set(self.page_text[self.index])
        # if the differential pachymetry map button was visible, forget the placement
        if self.diff_pachy_button == True:
            self.diff_pachy_button = False
            self.page[2].place_forget()

        # change the index to the other page
        self.index = abs(self.index-1)
        # if the other page does contrain the information of an AS-OCT scan, show this in the window
        if self.data[self.index] != None:
            self.update_different_scan()
            self.page[abs(self.index-1)].config(bg='white', relief=tk.RAISED)
            self.page[self.index].config(bg='light grey', relief=tk.SUNKEN)
        # if the other page was also empty, reset all states
        else:
            self.clear_window()
            self.page[abs(self.index-1)].config(bg='white', relief=tk.RAISED)
            self.dictionary = dict()
            self.idx = 0                        
            self.visible = True                 
            self.total_lines = 0                
            self.dashed_lines = tk.IntVar()     
            self.thickness_lines = tk.IntVar()  
            self.space = 10                     
            self.index = -1   

    def main_placement(self):
        """ Place all canvases and widgets in the normal position.
        """
        # start by clearing the window (the full screen B-scan or differential pachymetry map page was previously shown)
        self.clear_window()

        # place B-scan canvas and connect events to actions
        self.place_canvas('B_scan', 1)
        self.dictionary['B_scan'][0].mpl_connect("scroll_event", self.onscroll)
        self.dictionary['B_scan'][0].mpl_connect('button_press_event', self.onclick)

        # place pachymetry map canvas and connect events to actions
        self.place_canvas(f'Pachymetry_map{self.index}', 1)
        self.dictionary[f'Pachymetry_map{self.index}'][0].mpl_connect("scroll_event", self.onscroll)
        self.dictionary[f'Pachymetry_map{self.index}'][0].mpl_connect('button_press_event', self.onclick)

        # place thickness profile canvas and connect events to actions
        self.place_canvas('Thickness_profile',1)
        self.dictionary['Thickness_profile'][0].mpl_connect("scroll_event", self.onscroll)
        self.dictionary['Thickness_profile'][0].mpl_connect('button_press_event', self.onclick)

        # place widgets
        self.place_widget('Label_options', 1)
        self.place_widget('Check_9mm', 1)
        self.place_widget('Check_thickness_lines', 1)
        self.place_widget('Button_remove_data', 1)
        self.place_widget('Button_full_screen', 1)
        self.place_widget('Entry_pachy', 1)
        self.place_widget('Button_entry_pachy', 1)
        self.place_widget('Label_pachy_levels', 1)

    def diff_pachymetry_placement(self):
        """ Place all canvases and widgets in the positions for the differential pachymetry map page.
        """
        # start by clearing the window (the full screen B-scan or main screen page was previously shown)
        self.clear_window()

        # place canvases (no connections to events are necessary)
        self.place_canvas('Pachymetry_map0', 2)
        self.place_canvas('Pachymetry_map1', 2)
        self.place_canvas('Differential_pachymetry_map', 1)
        self.place_widget('Entry_pachy',2)
        self.place_widget('Button_entry_pachy', 2)
        self.place_widget('Label_pachy_levels', 2)
        self.place_widget('Entry_diff', 2)
        self.place_widget('Button_entry_diff', 2)
        self.place_widget('Label_diff_levels', 2)
        self.place_widget('Label_map', 1)

    def onscroll(self, event):
        """ Update class variables of states after scroll event.
        """
        # check if the page is not the differential pachymetry page
        if self.index <2:
            # if the user scrolled up, add one to idx that keeps track of the B-scan (except when the value is already max)
            if event.button == 'up':
                self.idx = min(self.idx+1, self.N-1)
            # if the user scrolled down, subtract one from idx that keeps track of the B-scan (except when the value is already 0)
            elif event.button == 'down':
                self.idx = max(self.idx-1, 0)
            self.update_same_scan()

    def onclick(self, event):
        """ Update class variables of states after left click event.
        """
        # check if the page is not the differential pachymetry page
        if self.index <2:
            # if the left mouse button was clicked, change the visibility
            if event.button == 1:
                if self.visible == True:
                    self.visible = False
                else:
                    self.visible = True
            self.update_same_scan()
        
    def update_same_scan(self):
        """ make changes to figures if the AS-OCT scan is the same (no switch between pages).
        """
        # retrieve data
        images = self.data[self.index][1]
        df = self.data[self.index][2]

        # change the image and title
        self.img.set_data(images[self.idx,:,:])
        self.ax1.set_title(f"{self.idx}/{self.N-1}", loc='right', fontsize=14, fontname='Open sans')
        # remove all lines that were plotted
        self.ax1.lines = []

        # plot the lines again if the visibility is set to true
        if self.visible == True:
            
            # plot the dashed lines if the box is checked
            if self.dashed_lines.get() == True:
                self.ax1.plot(2*[180], [0, self.image_size[0]], linestyle='--', color='w', linewidth = 2)
                self.ax1.plot(2*[780], [0, self.image_size[0]], linestyle='--', color='w', linewidth = 2)

            # plot the thickness lines
            if self.thickness_lines.get() == True:
                thickness_lines = df['thickness_lines'][self.idx][self.lim[0]:self.lim[1]]
                for i in np.arange(len(thickness_lines)):
                    if i%self.space == 0:
                        self.ax1.plot(thickness_lines[i][0], thickness_lines[i][1], color='cyan', linewidth = 1.5)
            
            # plot the network delineation
            self.ax1.plot(self.inside_x, df['inside_y_pred'][self.idx], color='r', linewidth = 2)
            self.ax1.plot(self.outside_x, df['outside_y_pred'][self.idx], color='r', linewidth = 2)

        # re-draw the delineated B-scan canvas
        self.canvas1.draw_idle()

        # change plotted line in thickness profile canvas plot
        self.ax3.lines = []
        px_radius = (self.lim[1]-self.lim[0])//2
        self.ax3.plot(np.arange(-px_radius, px_radius), df['thickness'][self.idx][self.lim[0]:self.lim[1]]*self.fact, color='k', linewidth = 1.5)
        self.canvas3.draw_idle()

        # change the red line that indicates the B-scan in the pachymetry
        self.pachymetry_map_changes(self.index)
        self.dictionary[f'Pachymetry_map{self.index}'][0].draw_idle()

    def update_different_scan(self):
        """ redraws all figures for changes after event.
        """
        # retrieve data
        images = self.data[self.index][1]
        df = self.data[self.index][2]

        # change the image and title
        self.img.set_data(images[self.idx,:,:])
        self.ax1.set_title(f"{self.idx}/{self.N-1}", loc='right', fontsize=14, fontname='Open sans')
        # remove all lines that were plotted
        self.ax1.lines = []

        # plot the lines again if the visibility is set to true
        if self.visible == True:
            
            # plot the dashed lines if the box is checked
            if self.dashed_lines.get() == True:
                self.ax1.plot(2*[180], [0, self.image_size[0]], linestyle='--', color='w', linewidth = 2)
                self.ax1.plot(2*[780], [0, self.image_size[0]], linestyle='--', color='w', linewidth = 2)

            # plot the thickness lines
            if self.thickness_lines.get() == True:
                thickness_lines = df['thickness_lines'][self.idx][self.lim[0]:self.lim[1]]
                for i in np.arange(len(thickness_lines)):
                    if i%self.space == 0:
                        self.ax1.plot(thickness_lines[i][0], thickness_lines[i][1], color='cyan', linewidth = 1.5)
            
            # plot the network delineation
            self.ax1.plot(self.inside_x, df['inside_y_pred'][self.idx], color='r', linewidth = 2)
            self.ax1.plot(self.outside_x, df['outside_y_pred'][self.idx], color='r', linewidth = 2)

        # redraw the delineated B-scan canvas
        self.canvas1.draw_idle()

        # change plotted line in thickness profile canvas plot
        self.ax3.lines = []
        px_radius = (self.lim[1]-self.lim[0])//2
        self.ax3.plot(np.arange(-px_radius, px_radius), df['thickness'][self.idx][self.lim[0]:self.lim[1]]*self.fact, color='k', linewidth = 1.5)
        self.canvas3.draw_idle()

        # change pachymetry map
        self.dictionary[f'Pachymetry_map{abs(self.index-1)}'][0].get_tk_widget().place_forget()
        self.placed_canvas.remove(f'Pachymetry_map{abs(self.index-1)}')
        self.pachymetry_map_changes(self.index)
        self.place_canvas(f'Pachymetry_map{self.index}',1)


if __name__ == '__main__':
    start_GUI()
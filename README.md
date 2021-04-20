# Corneal Pachymetry
This repository contains python code for corneal pachymetry in anterior segment optical coherence tomography (AS-OCT) scans as described in the paper **Corneal Pachymetry by AS-OCT after Descemet’s Membrane Endothelial Keratoplasty** by Heslinga *et al.* [1]. Analysis of the corneal thickness, estimated using a convolutional neural network (CNN), can be performed by means of a graphical user interface (GUI). The GUI allows for interactive inspection of corneal thinkness per radial slice, as well as for pachymetry maps and differential pachymetry maps of the cornea as a whole. The specific CNN provided in this repository is a small version of the *CNN with dimension reduction* described in [1], trained with only 50% of the filters to reduce the model size. The code for creating the full CNN with dimension reduction model is also available here.

[1] https://arxiv.org/abs/2102.07846

### Requirements
To install the required external python packages, run the following command:
```
pip install -r requirements.txt
```

### Folder structure
The repository has the following folder structure:
  
    ├── network 
    │   ├── network.h5 
    │   └── cnn_with_dim_red.py 
    ├── .gitignore 
    ├── README.md 
    ├── gui.py 
    └── requirements.txt  

- ```gui.py``` python code for graphical user interface.
- ```network.h5``` optimized convolutional neural network parameters for the CNN with dimension reduction architecture.
- ```network.py``` python code that creates the CNN with dimension reduction architecture in Tensorflow Keras.
- ```requirements.txt``` lists external python packages with version that are required.
- ```README.md``` explains the contents and the workings of the GUI.
- ```.gitignore``` lists files and folders to ignore when pushing from a local repository.

## Graphical User Interface
The explanation of the GUI below addresses all features and noteworthy details for analysis of AS-OCT images.

### Loading in the data
After executing ```gui.py```, a full screen window is created with in the top left corner two buttons to add AS-OCT scan images and one button in the top right corner to close the window. Click any of the two top left buttons to load in the data. Be aware that the GUI currently only supports 16 radial B-scan images (centered, 960x384 pixels) as input, which should be selected at once using SHIFT-click. The following naming convention for a single radial image is used: pt{*patient ID*}\_visit{*visit ID*}\_{*eye*}\_im{*image number*}.png, (e.g. pt001_visit07\_1\_im000.png). If the selected input images do not follow the specified naming convention, or if not exactly 16 images were selected, an error message is displayed. If a selection is accepted, the B-scan images are further processed, delineations are predicted using the CNN with dimension reduction, the thickness is measured, and finally the pachymetry map is made. These steps can take a number of seconds. Note that the GUI can temporarily become unresponsive during the prediction and processing stage. Loaded data can also be removed using the 'Remove data' button.

### Main screen
The main screen is divided into 4 sections: The B-scan (top left), the pachymetry map (top right), the thickness profile plot (bottom left), visualization settings (bottom right). Two AS-OCT scans of 16 images can be loaded at the same time. For more details on the thickness measurement and pachymetry mapping approach, we refer the user to the paper. Note that an artist impression of the B-scan was used.
![main_screen](https://user-images.githubusercontent.com/54849762/115051077-b44ef800-9edc-11eb-8b08-95efaa0ccb54.png)

#### B-scan
The B-scan is displayed together with the predicted delineation by the CNN in red. The user can use the scroll wheel of the mouse to scroll through the radial B-scan images. 
Left clicking makes the delineation invisible, which allows for better comparison with the true corneal interface.

#### Pachymetry map
The pachymatry map shows the thickness of the entire cornea from the B-scan thickness measurements. The average thicknesses of the regions are displayed as numeric values as well. When the user scrolls through the B-scan images, two opposite red lines start to rotate around the B-scan center, which indicates the radial image that is currently displayed.

#### Thickness profile plot
The thickness that is measured from the B-scan is also plotted in the thickness profile plot for more accurate interpretation. 
When the user scrolls through the B-scans, the thickness profile plot also changes to the corresponding B-scan image.

#### Visualisation options
The first setting that can be toggled on is 'Show 9mm diameter', which plots two white vertical dashed lines to indicate the 9mm diameter region. This aligns with the thickness plot below. The second option is 'Show thickness lines', which will show how the thickness is measured perpendicularly to the anterior interface for some of the measurement points. The last setting enables the user select the number of color levels for pachymetry map by selecting in a number between 1 and 200, followed by clicking the 'Apply entry' button.

### Additional screens
Under the B-scan, a button is positioned that will show the B-scan in full screen mode. It is still possible to cycle through all the B-scans from the AS-OCT scan in full screen with the scroll wheel. The user can also exit the full screen mode using a button under the B-scan. Furthermore, when two scans of the same patient and eye are loaded in (determined from the filenames), another button at the top becomes visible which directs the user to the differential pachymetry map page.
At the top left and bottom left are the first and second pachymetry map displayed, respectively. At the right in the center, the differential pachymetry map is shown. Note that it is important to load in the data in the correct order (latest scan must be loaded in secondly).
![differential_pachymetry](https://user-images.githubusercontent.com/54849762/115051089-b913ac00-9edc-11eb-8f47-bd9d3129c537.png)

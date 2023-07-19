## Topics:
* Phase 1
=> Build Data Collection Tool
=> Process and Visualize the collected data
=> Feature Extraction and Visualization
=> Understanding Data and Customization: Cleaning and Shaping
=> Nearest Neighbor Classifier and Data Scaling
=> K Nearest Neighbors Classifier | Calculate accuracy of different values of K
=> Decision Boundaries Computation

* Phase 2
=> Advanced Features
=> Higher Dimensions
=> Neural Networks
=> Data Cleaning
=> Clustering
=> Graphs

### High Level Overview:
1. It starts by creating a canvas for drawing to input data. people will draw the specified object and a node.js backend will store the person's name, id, session (timestamp), and all those input for drawing elements (mousedown move positions and multiple paths) for each subject. 

2. Data Visualization: google's data visual library is used to inspect (Feature extraction) those data by different perspective in 2 dimensional point. Also remove/flag the wrong data input

3. Data Behavior Inspection and Fine Tuning: A live draw canvas connected with the visualized data chart is used to further inspect how a single data behaves in realtime based on showing a point in the chart.

### Directory Overview:
* web : front-end part.
    - creator.html : starting point. Initialize the sketchpad.js by passing the container as id.
    - js/sketchpad.js provide the drawing canvas functionality. It calls draw.js. Note: ctx is the canvas context.

### Progress (Until):
* Code Implementation From Scratch:
    - 22.09 (round path join)
    - 31.47
* Copied Over:
* Watched and Noted:2.49.47
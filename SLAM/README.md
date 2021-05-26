# Simultaneous Localization and Mapping (SLAM) Project
<img src="https://github.com/ryabchenko-a/Udacity-CVND/blob/main/SLAM/images/robot_world.png" alt="Robot World" height="512"/>

In this project I've built a virtual robot that moves randomly and uses its not perfect sensor to sense the world (i.e. closest landmarks).

## Detailed description of the project

The project consists of 3 ipynb notebooks.

### [1. Robot Moving and Sensing.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/SLAM/1.%20Robot%20Moving%20and%20Sensing.ipynb)

This notebook starts the project by exploring the robot class and implementing its 2 main functions: moving and sensing. 
Robot's sensors are error-prone as well as its motion is not fully accurate. The robot class is also stored in [robot_class.py](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/SLAM/robot_class.py)

### [2. Omega and Xi, Constraints.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/SLAM/2.%20Omega%20and%20Xi%2C%20Constraints.ipynb)

This notebook presents the fundamentals of Graph SLAM - the constraint matrix Omega and it's supplement vector Xi. 

### [3. Landmark Detection and Tracking.ipynb](https://github.com/ryabchenko-a/Udacity-CVND/blob/main/SLAM/3.%20Landmark%20Detection%20and%20Tracking.ipynb)

The last notebook presents the actual implementation of Graph SLAM, that is building and updating Omega and Xi arrays and using them to sense the world.

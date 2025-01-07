# Koopman-based Robust Learning Control with Extended State Observer

[Paper]() | video | project
---

<img src="https://github.com/XinLang2019/koopman_active_learning/blob/master/doc/algorithm_frame.png" width="900"/>

- **Abstract**:
A key challenge in data-driven robot control is enabling robots to autonomously gather the most informative data during training while maintaining robust performance when deployed in new tasks or encountering unknown external disturbances. In this paper, we propose a robust active learning (RAL) control method designed to optimize data efficiency during model learning while ensuring robust and precise control during task execution. This approach integrates Koopman-based modeling with an active learning algorithm to enhance model learning efficiency, and an extended state observer (ESO)-assisted tracking control to ensure precise robot position control in the presence of unknown disturbances. The effectiveness of the proposed method is validated through various simulations and experiments, demonstrating significant improvements in data efficiency and robustness against unknown disturbances.

---

## Content
- [Introduction](#Introduction)
- [Instrall](#Instrall)
- [Using code](#Using)
- [Reference](#Reference)
- [BibTex Citation](#BibTex)
- [License](#License)

---

## Introduction
this repository is the code of koopman active learning with ESO. The code include the main stage of algorithm, such as `learning`, `lqr control` and `eso control`.

As our result, we can see the performance of algorithm in franka robot. we designed 2 experiments in simulation, and 3 experiments in real robot.

---

## Instrall
### system config
- System: Ubuntu 20.04 + ROS-noetic

### Step1: install libfrank

install dependence
```sh
sudo apt install build-essential cmake git libpoco-dev libeigen3-dev
```

```sh
git clone --recursive https://github.com/frankaemika/libfranka --branch 0.10.0 # only for FR3
```

```sh
cd libfranka
```

```sh
mkdir build 
cd build 
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. 
cmake --build .
```

this order is `options`
```sh
cpack -G DEB
sudo dpkg -i libfranka*.deb
```

### Step2: install this package
```sh
git clone https://github.com/XinLang2019/koopman_active_learning.git
```

```sh
shcd koopman_active_learning 
```

```sh
source /opt/ros/noetic/setup.sh
```

```sh
catkin_init_workspace src
```

```sh
rosdep install --from-paths src --ignore-src --rosdistro noetic -y --skip-keys libfranka 
```
if this step have some error like `rosdep not install`, You just need to follow the prompts to install the corresponding package.Then `build it`

```sh
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DFranka_DIR:PATH=/path/to/libfranka/build  
``` 
notice `/path/to/` is your `libfrank` path.

## Using code
- Firstly, open **one terminal** launch franka simulation environment(gazebo)
```sh
roslaunch franka_gazebo panda.launch controller:=joint_effort_example_controller rviz:=true
```

- you can run active learning `koopman_active_learning/src/koopman_learning.cpp`, by:
```sh
rosrun koopman_active_learning koopman_learning_node
```

- next, you can run the `RAL` algorithm `koopman_active_learning/src/koopman_lqr_eso.cpp`, you can open **other terminal**:
```sh
rosrun koopman_active_learning koopman_lqr_eso_node
```
- or you can run the experiment of gripping, you can run:
```sh
rosrun koopman_active_learning koopman_lqr_gripping_node
```

### Notice 
In ESO node `koopman_active_learning/src/koopman_lqr_eso.cpp`, you can change the variable `method (lines 217)` with `false` or `true` to shift the `baseline` or `proposed` method, and the variable `traj_index (lines 218)` to change the task trajectories.

when you change the code, you **must run** `catkin_make` to rebuild your code.

## Reference
The active learning part of code reference: https://github.com/i-abr/active-learning-koopman.git

## BibTex Citation


## License
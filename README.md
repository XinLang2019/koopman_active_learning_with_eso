# Koopman-based Robust Learning Control with Extended State Observer

![](https://github.com/XinLang2019/koopman_active_learning/blob/master/doc/algorithm_frame.png) <!-- 可选：项目 logo 或状态徽章 -->

- Abstract:
A key challenge in data-driven robot control is enabling robots to autonomously gather the most informative data during training while maintaining robust performance when deployed in new tasks or encountering unknown external disturbances. In this paper, we propose a robust active learning (RAL) control method designed to optimize data efficiency during model learning while ensuring robust and precise control during task execution. This approach integrates Koopman-based modeling with an active learning algorithm to enhance model learning efficiency, and an extended state observer (ESO)-assisted tracking control to ensure precise robot position control in the presence of unknown disturbances. The effectiveness of the proposed method is validated through various simulations and experiments, demonstrating significant improvements in data efficiency and robustness against unknown disturbances.

---

## Content
- [Introduction](#Introduction)
- [Instrall](#Instrall)
- [Using code](#Using)
- [BibTex Citation](#BibTex)
- [License](#License)

---

## Introduction
this repository is the code of koopman active learning with ESO. The code include the main stage of algorithm, such as `learning`, `lqr control` and `eso control`.

As our result, we can see the performance of algorithm in franka robot. we designed 2 experiments in simulation, and 3 experiments in real robot.

- at koopman learning stage, we can see the active learning trajectory and without active learning trajectory:

<p float="center">
  <img src="https://github.com/XinLang2019/koopman_active_learning/blob/master/doc/dynamic_learning_1.png" width="35%" />
  <img src="https://github.com/XinLang2019/koopman_active_learning/blob/master/doc/dynamic_learning_2.png" width="50%" />
</p>

---

## Instrall
### 前置要求
- 操作系统：Windows/Linux/macOS
- Python 版本：>= 3.8

### 安装步骤
1. 克隆项目代码：
   ```bash
   git clone https://github.com/your_username/your_project.git
   cd your_project

## Using code


## BibTex Citation

## License
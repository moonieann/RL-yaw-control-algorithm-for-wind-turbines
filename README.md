# RL-based yaw control algorithm for wind turbines

## Abstract
Yaw misalignment, measured as the difference between the wind direction and the nacelle position of a wind turbine, has consequences on the power output, the safety and the lifetime of the turbine and its wind park as a whole. We develop a yaw control algorithm that uses a Reinforcement-Learning-trained agent to minimise yaw misalignment and optimally reallocate yaw resources, prioritising high-speed segments, while keeping yaw usage low. We compare our algorithm to the conventional active yaw control algorithm on a real-world dataset obtained from turbine logs of a REpower MM82 2MW turbine. The algorithm decreased the yaw misalignment by 5.5% and 11.2% on the two 2.7 hours datasets compared to the traditional yaw control algorithm. The average net energy gain obtained was 0.31% and 0.33% compared to the traditional yaw control algorithm.  On a single 2MW turbine, this amounts to a 1.5k-2.5k euros annual gain, which sums up to very significant profit over an entire wind park.

## Install requirements

```
$ pip install -r requirements.txt
```

## Run Experiment (training of RLYCA and test against CYCA-S and CYCA-L)

Steady wind conditions :
```
$ python3 steady_script.py
```
Variable wind conditions :
```
$ python3 variable_script.py
```

 





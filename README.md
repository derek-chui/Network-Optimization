# Network Optimization

This repository contains a collection of projects regarding efficient user clustering in NOMA (Non Orthogonal Multiple Access) 6G Networks

Click on the arrows to select and expand each project for more details

## Contents

### Current Projects (In Progress)
- [Semantic Q Learning NOMA](#semantic-q-learning-noma)
- [Pinch Antenna Assisted NOMA](#pinch-antenna-assisted-noma)

### Completed Projects
- [Semantic Greedy NOMA](#semantic-greedy-noma)

## Semantic Q Learning NOMA 

- **Status**: In Progress 
- **Summary**: Q Learning as a reinforcement learning algorithm & improved semantic model for NOMA. Compared with previous algorithms from [Semantic Greedy NOMA](#semantic-greedy-noma) to show significant improvements in network utility.
- **Latest Viable Product**: [CSEN193-Report](https://drive.google.com/file/d/1Wj3fvNztdjBkblzAmVwG4N0A-NX2bxKQ/view?usp=sharing)
- **Folder**: [q-learning](./q-learning/)

<details>
    <summary>Submissions</summary>
    
**Below are the submissions (conferences, symposiums, grants, awards, etc.) derived from this work:**
<!-- - [2026 Kuehler Undergraduate Research Award]()
- [2026 SCU School of Engineering Research Showcase]()
- [2026 Helene Lafrance Library Undergraduate Research Award]() -->
- [2025 Hackworth Applied Ethics Research Grant](https://drive.google.com/file/d/1fAArEWDzZdylOofSUgRrsInHTT-JssLH/view?usp=sharing) (Rejected)
- [2025 BayLearn Machine Learning Symposium](https://drive.google.com/file/d/1sKTumPeglMpADQQpm-wT4cGTqHteR7Kh/view?usp=sharing) (Rejected)

</details>

<details>
    <summary>Simulations</summary>
    
**Below are the simulations used for this work:**

- [NOMA3Q](./q-learning/NOMA3Q.m): Groups users in triplets and returns utilities with q learning
- [NOMA3Q2](./q-learning/NOMA3Q2.m): Groups users in triplets with new semantic model and returns utilities with q learning

</details>

<details>
    <summary>Resources</summary>
    
**Below are the resources used for this work:**

- [CSEN193-Report](https://drive.google.com/file/d/1Wj3fvNztdjBkblzAmVwG4N0A-NX2bxKQ/view?usp=sharing)
- [Q-Learning Slides](https://docs.google.com/presentation/d/1PHditpACiUvYX8aJAIAcmyD-i9RLTeeeETR5KbQPe-E/edit?usp=sharing)
- [Q-Learning Research Notes](https://docs.google.com/document/d/18udG3UeT6BD7QErJse2ASpI6ao0Ye07AF26nnO2YBTM/edit?usp=sharing)

</details>

## Pinch Antenna Assisted NOMA

- **Summary**: Enhanced NOMA using strategically placed pinch antennas to amplify user channels based on proximity. Improving effective gains over baseline, which enables scalable, antenna assisted user grouping for future 6G networks.
- **Status**: On Hold until Spring 2026
- **Latest Viable Product**: [simulation](./pinch-antenna/simulation.m)
- **Folder**: [pinch-antenna](./pinch-antenna/)

<details>
    <summary>Submissions</summary>
    
**Below are the submissions (conferences, symposiums, grants, awards, etc.) derived from this work:**

</details>

<details>
    <summary>Simulations</summary>
    
**Below are the simulations used for this work:**

- [simulation](./pinch-antenna/simulation.m): Maximize sum rate based on users and antennas
- [graphs](./pinch-antenna/graphs.m): Evaluates antenna activation patterns and plotting average sum rates and usage

</details>


<details>
    <summary>Resources</summary>
    
**Below are the resources used for this work:**

- [Pinch Antenna Research Notes](https://docs.google.com/document/d/1huzSNCxg2J__nn4J5MR6-25t5MMbg6vueq4OWNWkKiU/edit?usp=sharing)

</details>

## Semantic Greedy NOMA

- **Summary**: Greedy algorithm & simple semantic model for NOMA. Compared with existing algorithms drawn from other papers to show significant improvements in network utility.
- **Status**: Completed & Improved in [Semantic Q Learning NOMA](#semantic-q-learning-noma)
- **Final Product**: [IEEE CCNC 2026 SG-NOMA](./SG-NOMA/IEEE%20CCNC%202026%20SG-NOMA.pdf)
- **Folder**: [SG-NOMA](./SG-NOMA/)

<details>
    <summary>Submissions</summary>
    
**Below are the submissions (conferences, symposiums, grants, awards, etc.) derived from this work:**

- [2026 IEEE Consumer Communications & Networking Conference](https://drive.google.com/file/d/1bnmEY68AvoT-BOUoi62nJNvAtt0XLwHg/view?usp=sharing) (Rejected as regular paper but accepted as a poster)
- [2026 IEEE Consumer Communications & Networking Conference (Poster)]([./SG-NOMA/NOMA3S/IEEE%20CCNC%202026%20SG-NOMA%20(Short).pdf](https://drive.google.com/file/d/1TZ-hggjblw8D4OI7lmwQi5GvmUcgvjQ7/view?usp=sharing)) (Withdrawn)

</details>

<details>
    <summary>Simulations</summary>
    
**Below are the simulations used for this work:**

- [randPoints](./SG-NOMA/randPoints.m): Generates random points across a 2D plane
- [bruteForce](./SG-NOMA/bruteForce.m): Baseline protocol, returns the best possible utility
- [NOMA2](./SG-NOMA/NOMA2.m): Pairs users and returns utilities for all researched algorithms
- [NOMA3](./SG-NOMA/NOMA3.m): Groups users in triplets and returns utilities for all researched algorithms
- [etaRayleigh](./SG-NOMA/etaRayleigh.m): Utilities as a function of path loss exponent and Rayleigh fading
- [powerNoise](./SG-NOMA/powerNoise.m): Utilities as a function of power P and noise N0
- [powerNoiseEtaRayleigh](./SG-NOMA/NOMA3S/powerNoiseEtaRayleigh.m): Utilities as a function of several variables (used for shorter version)
- [NOMA23](./SG-NOMA/NOMA3S/NOMA23.m): Grouping results for both pairs and triplets (used for shorter version)

</details>

<details>
    <summary>Resources</summary>
    
**Below are the resources used for this work:**

- [NOMA Slides](https://docs.google.com/presentation/d/1_N1oKkR_PmWWJWkS9RF0X-JVHOiJuH3OqhkIK069pV0/edit?usp=sharing)
- [SG-NOMA Research Notes](https://docs.google.com/document/d/14G8pNsJsSaJc02iIsvGAqQGKgUyCtUJMqTkqEhJl50w/edit?usp=sharing)

</details>

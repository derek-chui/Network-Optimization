# Network Optimization

Efficient user clustering in NOMA (Non Orthogonal Multiple Access) 6G Networks.

Click on the arrows to select and expand each project for more details!

## Contents

### Current Projects (In Progress)
- [Semantic Q Learning NOMA](#semantic-q-learning-noma)

### Completed Projects
- [Semantic Greedy NOMA](#semantic-greedy-noma)

### Future Projects
- [Pinch Antenna Assisted NOMA](#pinch-antenna-assisted-noma)

## Semantic Q Learning NOMA 

- **Status**: In Progress 
- **Summary**: Q Learning as a reinforcement learning algorithm & improved semantic model for NOMA. Consistent gains in network utility and scalability for future 6G wireless systems. Compared with previous algorithms from [Semantic Greedy NOMA](#semantic-greedy-noma) to show significant improvements in network utility.
- **Latest Viable Product**: [State of the Field Essay](https://drive.google.com/file/d/1C5jopMzgl9t2hgAJOYhTnA18xQ7LQbZU/view?usp=sharing)
- **Folder**: [q-learning](./q-learning/)

<details>
    <summary>Submissions</summary>
    
**Below are the submissions (conferences, symposiums, grants, awards, etc.) derived from this work:**

<!-- - [SCU School of Engineering Research Showcase]() -->
<!-- - [Helene Lafrance Library Undergraduate Research Award]() -->
<!-- - [SQ-NOMA Paper]() -->
- [Hackworth Applied Ethics Research Grant](https://drive.google.com/file/d/1C5jopMzgl9t2hgAJOYhTnA18xQ7LQbZU/view?usp=sharing) (Submission)
- [BayLearn Machine Learning Symposium](https://drive.google.com/file/d/1sKTumPeglMpADQQpm-wT4cGTqHteR7Kh/view?usp=sharing) (Submission)

</details>

<details>
    <summary>Resources</summary>
    
**Below are the resources used for this work:**

- [State of the Field Essay](https://drive.google.com/file/d/1C5jopMzgl9t2hgAJOYhTnA18xQ7LQbZU/view?usp=sharing)
- [State of the Field Poster](https://drive.google.com/file/d/1iz9HXbXjWAm1whCISTmGGljaUdx4K9RJ/view?usp=sharing)
- [Q-Learning Slides](https://docs.google.com/presentation/d/1PHditpACiUvYX8aJAIAcmyD-i9RLTeeeETR5KbQPe-E/edit?usp=sharing)
- [Q-Learning Research Notes](https://docs.google.com/document/d/18udG3UeT6BD7QErJse2ASpI6ao0Ye07AF26nnO2YBTM/edit?usp=sharing)

</details>

<details>
    <summary>Simulations</summary>
    
**Below are the simulations used for this work:**

- [NOMA3Q2](./q-learning/NOMA3Q2.m): Groups users in triplets, compares utilities with q learning
- [NOMA2Q2](./q-learning/NOMA2Q2.m): Groups users in pairs, compares utilities with q learning
- [powerNoiseEtaRayleigh.m](./q-learning/powerNoiseEtaRayleigh.m): Compares utilities across multiple algorithms under various conditions
- [NOMA3Q](./q-learning/NOMA3Q.m): Groups users in triplets with old semantic model, compares utilities with q learning

</details>

















## Semantic Greedy NOMA

- **Summary**: Greedy algorithm & simple semantic model for NOMA that outperforms existing baselines in network utility. Strong foundation later extended with reinforcement learning in [Semantic Q Learning NOMA](#semantic-q-learning-noma).
- **Status**: Completed
- **Final Product**: [Semantic Utility Aware User Grouping for 6G NOMA Networks](https://drive.google.com/file/d/1bnmEY68AvoT-BOUoi62nJNvAtt0XLwHg/view?usp=sharing)
- **Folder**: [SG-NOMA](./SG-NOMA/)

<details>
    <summary>Submissions</summary>
    
**Below are the submissions (conferences, symposiums, grants, awards, etc.) derived from this work:**

- [IEEE Consumer Communications & Networking Conference (Poster)](https://drive.google.com/file/d/1TZ-hggjblw8D4OI7lmwQi5GvmUcgvjQ7/view?usp=sharing) (Accepted)
- [IEEE Consumer Communications & Networking Conference](https://drive.google.com/file/d/1bnmEY68AvoT-BOUoi62nJNvAtt0XLwHg/view?usp=sharing) (Submission)

</details>

<details>
    <summary>Resources</summary>
    
**Below are the resources used for this work:**

- [SG-NOMA Slides](https://docs.google.com/presentation/d/1_N1oKkR_PmWWJWkS9RF0X-JVHOiJuH3OqhkIK069pV0/edit?usp=sharing)
- [SG-NOMA Research Notes](https://docs.google.com/document/d/14G8pNsJsSaJc02iIsvGAqQGKgUyCtUJMqTkqEhJl50w/edit?usp=sharing)

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


















## Pinch Antenna Assisted NOMA

- **Summary**: Enhanced NOMA using strategically placed pinch antennas to amplify user channels based on proximity. Potential for scalable user grouping and improved sum rates in future 6G networks.
- **Status**: On Hold until Spring 2026
- **Latest Viable Product**: [simulation](./pinch-antenna/simulation.m)
- **Folder**: [pinch-antenna](./pinch-antenna/)

<details>
    <summary>Submissions</summary>
    
**Below are the submissions (conferences, symposiums, grants, awards, etc.) derived from this work:**
<!-- - [Kuehler Undergraduate Research Award]() -->
<!-- - [Pinch Antenna Paper]() -->

</details>

<details>
    <summary>Resources</summary>
    
**Below are the resources used for this work:**

- [Pinch Antenna Research Notes](https://docs.google.com/document/d/1huzSNCxg2J__nn4J5MR6-25t5MMbg6vueq4OWNWkKiU/edit?usp=sharing)

</details>

<details>
    <summary>Simulations</summary>
    
**Below are the simulations used for this work:**

- [simulation](./pinch-antenna/simulation.m): Maximize sum rate based on users and antennas
- [graphs](./pinch-antenna/graphs.m): Evaluates antenna activation patterns and plotting average sum rates and usage

</details>

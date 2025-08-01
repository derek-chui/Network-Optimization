# Network Optimization

This repository contains a collection of projects regarding efficient user clustering in NOMA 6G Networks

Below are all the resources and files used for the following publications:

<details>
    <summary>IEEE CCNC 2026: "Semantic Utility Driven User Pairing and Triplet Grouping for NOMA Networks"</summary>

This paper looks into optimal algorithms for user pairing and triplet grouping in Non-Orthogonal Multiple Access (NOMA) wireless systems, aiming to maximize total utility by unifying channel characteristics with semantic relevance. As 6G wireless systems move toward task-oriented communication, it becomes increasingly important to consider the meaning and importance of user data in resource allocation. We evaluate several existing algorithms including brute force, Hungarian, and greedy approaches under a utility model that incorporates semantic value. Our study show that many traditional algorithms, designed without semantics in mind, perform suboptimally in this setting. Therefore, we propose a greedy algorithm called Semantic Greedy NOMA (SG-NOMA) that considers both channel diversity and semantic value, and demonstrate through simulations that it closely approximates brute force performance with significantly lower complexity. These findings highlight the importance of integrating semantic considerations into user grouping strategies for 6G wireless NOMA deployments.

</details>

<details>
    <summary>BayLearn 2025 Call for Abstracts: "Semantic utility driven NOMA networks & reinforcement learning."</summary>

Non-Orthogonal Multiple Access (NOMA) systems allow simultaneous communication among users with varying channel conditions, maximizing spectral efficiency via power-domain multiplexing. Traditional user pairing methods, such as greedy algorithms, optimize based on distance and fading, but overlook the content-level importance of the transmitted data. In this work, we propose a reinforcement learning framework for semantic-aware user pairing, where a Q-learning agent learns to group users by jointly considering physical channel conditions and the semantic value of their data. Users transmitting more meaningful or application-critical information are prioritized in pairing, leading to improved network performance from both spectral and content perspectives. We simulate a 10-user environment with randomized channel conditions and semantic priorities, and train the agent over 1000+ episodes. Preliminary results show that the learned policy captures pairing patterns similar to greedy baselines while offering greater adaptability for dynamic user and traffic profiles. This approach reflects a key design goal of 6G networks, to intelligently allocate resources based on both signal quality and data importance, and offers a path toward maximizing overall utility in future wireless systems.

</details>

<p><strong>Both of these are submitted and currently being reviewed.</strong></p>

The initial outline and draft can be found here:

[NOMA Slidedeck](https://docs.google.com/presentation/d/1_N1oKkR_PmWWJWkS9RF0X-JVHOiJuH3OqhkIK069pV0/edit?usp=sharing)

See the paper notes for insights on some of the papers referenced:

[Paper Notes](https://docs.google.com/document/d/14G8pNsJsSaJc02iIsvGAqQGKgUyCtUJMqTkqEhJl50w/edit?tab=t.0)

The MATLAB algorithms used to develop the simulations can be viewed below:

- [randPoints](./randPoints.m): Generates random points across a 2D plane
- [bruteForce](./bruteForce.m): Baseline protocol, returns the best possible utility
- [NOMA2](./NOMA2.m): Pairs users and returns utilities for all researched algorithms
- [NOMA3](./NOMA3.m): Groups users in triplets and returns utilities for all researched algorithms
- [etaRayleigh](./etaRayleigh.m): Utilities as a function of path loss exponent and Rayleigh fading
- [powerNoise](./powerNoise.m): Utilities as a function of power P and noise N0

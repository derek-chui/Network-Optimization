# Network Optimization

This repository contains a collection of projects regarding efficient user clustering in NOMA 6G Networks

Below are all the resources and files used for the following submissions:

- IEEE CCNC 2026: "Semantic Utility Driven User Pairing and Triplet Grouping for NOMA Networks"
- BayLearn 2025 Call for Abstracts: "Semantic utility driven NOMA networks & reinforcement learning."

The initial outline and draft can be found here:

[NOMA Slidedeck](https://docs.google.com/presentation/d/1_N1oKkR_PmWWJWkS9RF0X-JVHOiJuH3OqhkIK069pV0/edit?usp=sharing)

See the paper notes for insights on some of the papers referenced:

[Paper Notes](https://docs.google.com/document/d/14G8pNsJsSaJc02iIsvGAqQGKgUyCtUJMqTkqEhJl50w/edit?tab=t.0)

The MATLAB algorithms used to develop the simulations can be viewed below:

- [randPoints](./randPoints.m): Generates random points across a 2D plane
- [bruteForce](./bruteForce.m): Baseline protocol, returns the best possible utility
- [NOMA2](./NOMA2.m): Pairs users and returns utilities for all researched algorithms
- [NOMA3](./NOMA3.m): Groups users in triplets and returns utilities for all researched algorithms

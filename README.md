# Network-Optimization

Non Orthogonal Multiple Access is a wireless communication technique used in 5G to allow multiple users to share the same time and frequency resources, but with different power levels.

NOMA works best when a strong (close) user is paired with a weak (far) user. So our goal is to makes pairs from a set of users with a max total distance when added up.

[NOMA Slidedeck](https://docs.google.com/presentation/d/1_N1oKkR_PmWWJWkS9RF0X-JVHOiJuH3OqhkIK069pV0/edit?usp=sharing)

- Overview: Generates random points, makes pairs with max total distance using MATLAB.
- First implemented brute force O(n!!) as a baseline. Goal is to get brute force results in O(1) time.
- Working on implementing the many types of NOMA O(nlogn) from various research papers. This is much faster, but worse results than brute force.
- [NOMA Documentation](https://docs.google.com/document/d/14G8pNsJsSaJc02iIsvGAqQGKgUyCtUJMqTkqEhJl50w/edit?tab=t.0)
- Later will look into pattern recognition.
- Below are the descriptions for each file

## randPoints
- input points: 10 for numPoints, xRange, yRange if no input, otherwise must have 3, numPoints must be even
- make random points
- make graph with points

![Screenshot 2025-03-27 at 11 42 18 PM](https://github.com/user-attachments/assets/dfbb2514-e76c-43be-807a-58a706660a95)

## bruteForce O(n!!)
- make 10 random points
- sort these 10 points relative to origin (1 closest)
- make brute force pairing O(n!!) with points
- make comparison pairing O(1) with points
- show brute force pairs on graph
- show results in command window
- HELPER FUNCTIONS
- tot dist for this pairing set
- generates all unique pairings (brute force) recursively

![Screenshot 2025-04-02 at 10 50 41 PM](https://github.com/user-attachments/assets/97da6207-e9d2-405a-8aba-593c031a6759)
![Screenshot 2025-04-02 at 10 51 17 PM](https://github.com/user-attachments/assets/40f38a49-3171-4887-9beb-e5b4ebd6b69b)

## NOMA
Added:
- all pairing functions
- all NOMA helper functions

Removed:
- make comparison pairing O(1) with points

![Screenshot 2025-05-04 at 12 50 10 PM](https://github.com/user-attachments/assets/ce6f9004-df51-4c1e-a69c-0f557cd43462)
![Screenshot 2025-05-04 at 12 50 27 PM](https://github.com/user-attachments/assets/6b2a7b09-2f8d-4d08-8147-25e767053b04)

The following papers are referenced in making these algorithms:
- D-NLUPA: Dynamic User Clustering and Power Allocation for Uplink and Downlink Non-Orthogonal Multiple Access (NOMA) Systems
- D-NOMA: New User Grouping Scheme for Better User Pairing in NOMA Systems
- MUG: Multi-user Grouping Algorithm in Multi-carrier NOMA System
- LCG&DEC: Performance of Deep Embedded Clustering and Low-Complexity Greedy Algorithm for Power Allocation in NOMA-Based 5G Communication
- Hungarian&JV: Optimal user pairing using the shortest path algorithm

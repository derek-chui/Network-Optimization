# Network-Optimization

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
%- all pairing functions
%- all NOMA helper functions

Removed:
- make comparison pairing O(1) with points

![Screenshot 2025-04-27 at 5 14 08 PM](https://github.com/user-attachments/assets/4c7350d3-acc7-4cba-88f0-21cbdfbf3f2c)
![Screenshot 2025-04-27 at 5 14 38 PM](https://github.com/user-attachments/assets/3ec9d92a-bf4f-49e1-916d-eb7a5f7e859d)

The following papers are referenced in making these algorithms:
- D-NLUPA: Dynamic User Clustering and Power Allocation for Uplink and Downlink Non-Orthogonal Multiple Access (NOMA) Systems
- D-NOMA: New User Grouping Scheme for Better User Pairing in NOMA Systems
- MUG: Multi-user Grouping Algorithm in Multi-carrier NOMA System
- LCG_DEC: Performance of Deep Embedded Clustering and Low-Complexity Greedy Algorithm for Power Allocation in NOMA-Based 5G Communication

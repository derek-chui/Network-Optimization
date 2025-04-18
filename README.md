# Network-Optimization

Generates random points, makes pairs with max total distance using MATLAB.
Currently implemented brute force O(n!!) comparing it with default O(1) and NOMA(nlogn) sets.
Working on implementing and comparing it with greedy O(n^3).

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

## NOMA O(nlogn)
- D-NOMA and D-NLUPA pairings O(nlogn)
- D-NOMA and D-NLUPA funcs

![Screenshot 2025-04-10 at 8 49 09 PM](https://github.com/user-attachments/assets/e9e0b290-4219-4a70-a545-3a340577bf78)
![Screenshot 2025-04-10 at 8 49 43 PM](https://github.com/user-attachments/assets/d9226d26-2d58-4a40-a5c4-5dbd48ef4bf3)

The following papers are referenced in making these algorithms:
- D-NLUPA: Dynamic User Clustering and Power Allocation for Uplink and Downlink Non-Orthogonal Multiple Access (NOMA) Systems
- D-NOMA: New User Grouping Scheme for Better User Pairing in NOMA Systems

## Next: Greedy O(n^3)
- greedy pairings O(n^3)
- greedy funcs

## Next: Cache O(1)

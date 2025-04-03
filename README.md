# Network-Optimization

Generates random points, makes pairs with max total distance using MATLAB.
Currently implemented brute force O(n!!) comparing it with default O(1) sets.
Comparing how much more max total distance the brute force O(n!!) set are compared to comparison O(1) sets.

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

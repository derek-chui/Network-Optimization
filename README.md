# Network-Optimization

Non Orthogonal Multiple Access is a wireless communication technique used in 5G to allow multiple users to share the same time and frequency resources, but with different power levels.

NOMA works best when a strong (close) user is paired with a weak (far) user. So our goal is to makes pairs from a set of users with a max total distance when added up.

- Overview: Generates random points, makes pairs with max total distance using MATLAB.
- First implemented brute force O(n!!) as a baseline. Goal is to get brute force results in O(1) time.
- Then implemented the many types of NOMA from various research papers. This is much faster, but worse results than brute force.
- See the paper notes for insights on the papers referenced:

[Paper Notes](https://docs.google.com/document/d/14G8pNsJsSaJc02iIsvGAqQGKgUyCtUJMqTkqEhJl50w/edit?tab=t.0)

- Added triplets instead of pairs. Then added random weights to each point. Instead of distances, get scores (distance + weight difference).
- Specifics can be found on the slidedeck:

[NOMA Slidedeck](https://docs.google.com/presentation/d/1_N1oKkR_PmWWJWkS9RF0X-JVHOiJuH3OqhkIK069pV0/edit?usp=sharing)

- Below are the descriptions for each file

## randPoints
![Screenshot 2025-03-27 at 11 42 18 PM](https://github.com/user-attachments/assets/dfbb2514-e76c-43be-807a-58a706660a95)

## bruteForce O(n!!)
![Screenshot 2025-04-02 at 10 50 41 PM](https://github.com/user-attachments/assets/97da6207-e9d2-405a-8aba-593c031a6759)
![Screenshot 2025-04-02 at 10 51 17 PM](https://github.com/user-attachments/assets/40f38a49-3171-4887-9beb-e5b4ebd6b69b)

## NOMA
![Screenshot 2025-05-10 at 9 56 57 PM](https://github.com/user-attachments/assets/08f6ed77-64fd-4cff-a0fd-7abebda8ed5b)
![Screenshot 2025-05-10 at 9 57 09 PM](https://github.com/user-attachments/assets/64ca3819-bde8-41c6-9774-dcf9acfd2b66)
![Screenshot 2025-05-10 at 9 57 16 PM](https://github.com/user-attachments/assets/5e10b689-1a3c-4442-9501-169822a71275)

The following papers are referenced in making these algorithms:
- D-NLUPA: Dynamic User Clustering and Power Allocation for Uplink and Downlink Non-Orthogonal Multiple Access (NOMA) Systems
- D-NOMA: New User Grouping Scheme for Better User Pairing in NOMA Systems
- MUG: Multi-user Grouping Algorithm in Multi-carrier NOMA System
- LCG&DEC: Performance of Deep Embedded Clustering and Low-Complexity Greedy Algorithm for Power Allocation in NOMA-Based 5G Communication
- Hungarian&JV: Optimal user pairing using the shortest path algorithm

## NOMA3S
![Screenshot 2025-06-14 at 2 21 09 AM](https://github.com/user-attachments/assets/1942b115-fcf5-49d8-af99-6522bc70cc9e)
![Screenshot 2025-06-14 at 2 21 42 AM](https://github.com/user-attachments/assets/57f14ffc-2562-48aa-9043-648af42cf8dc)
![Screenshot 2025-06-14 at 2 22 17 AM](https://github.com/user-attachments/assets/5cc75b0c-8514-4666-b011-56763c747482)

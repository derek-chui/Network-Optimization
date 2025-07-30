# Network-Optimization

Below are all the resources and files used for the the following submissions:

- IEEE CCNC 2026: "Semantic Utility Driven User Pairing and Triplet Grouping for NOMA Networks"
- BayLearn 2025 Call for Abstracts: "Semantic Aware User Pairing for 6G NOMA Networks Using Reinforcement Learning"

Non Orthogonal Multiple Access is a wireless communication technique used in 5G to allow multiple users to share the same time and frequency resources, but with different power levels. NOMA works best when a strong (close) user is paired with a weak (far) user. So our goal is to makes pairs from a set of users with max total utility.

The paper proposes a semantic aware greedy algorithm for user pairing and triplet grouping in NOMA. By integrating semantic weights into the utility function, the algorithm prioritizes both channel diversity and information importance. It is compared against several other algorithms, with varying complexity, demonstrating that it closely approximates optimal performance while achieving significantly lower computational complexity. Extensive simulations show the effectiveness of semantic integration in enhancing communication efficiency and utility in realistic network conditions.

- A brief outline can be found on the slidedeck:

[NOMA Slidedeck](https://docs.google.com/presentation/d/1_N1oKkR_PmWWJWkS9RF0X-JVHOiJuH3OqhkIK069pV0/edit?usp=sharing)

- See the paper notes for insights on some of the papers referenced:

[Paper Notes](https://docs.google.com/document/d/14G8pNsJsSaJc02iIsvGAqQGKgUyCtUJMqTkqEhJl50w/edit?tab=t.0)

- The MATLAB algorithms used to develop the simulation can be found in this repository.

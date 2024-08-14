# Causal Inference

Causal discovery and causal inference are essential techniques in data science, statistics, and artificial intelligence for understanding and modeling cause-and-effect relationships. Let's break down these concepts from the ground up.

## 1. **Introduction to Causal Inference**

### **What is Causal Inference?**

Causal inference is the process of determining whether and how one variable (the cause) affects another variable (the effect). Unlike correlation, which measures the statistical association between variables, causation implies a directional relationship where changes in one variable lead to changes in another.

### **Why is Causal Inference Important?**

- **Predicting outcomes**: Knowing the causal factors can help predict future outcomes.
- **Understanding mechanisms**: It helps in understanding the underlying mechanisms behind observed phenomena.
- **Making decisions**: Causal knowledge is crucial for decision-making, particularly in policy-making, medicine, and economics.

## 2. **Basic Concepts in Causal Inference**

### **Random Variables and Causality**

Consider random variables $X$ and $Y$. If $X$ causes $Y$, we denote this as $X \rightarrow Y$. This means that changes in $X$ lead to changes in $Y$.

### **Directed Acyclic Graphs (DAGs)**

Causal relationships can be represented using Directed Acyclic Graphs (DAGs), where nodes represent variables, and edges represent causal influences. For example, the graph $X \rightarrow Y \rightarrow Z$ indicates that $X$ causes $Y$, which in turn causes $Z$.

### **Interventions**

An intervention refers to actively changing a variable to see how it affects another variable. This is often represented using the do-operator $\text{do}(X=x)$, which denotes setting $X$ to a specific value $x$ and observing the effects.

## 3. **Mathematical Foundations**

### **Causal Models**

A causal model consists of:
- **Structural equations**: These express each variable as a function of its causes and possibly some noise. For example, $Y = f(X, U_Y)$, where $U_Y$ is a noise term.
- **Causal diagrams**: Represent the structural equations graphically.

### **Conditional Independence**

In a causal graph, two variables $X$ and $Y$ are conditionally independent given $Z$ if knowing $Z$ makes $X$ and $Y$ independent of each other.

Mathematically, $X \perp\!\!\!\perp Y \mid Z$.

### **Do-Calculus**

To calculate the effect of interventions, we use Pearl's do-calculus. The key idea is to adjust the joint probability distribution to reflect the intervention.

#### **Rule 1: Ignoring Observations**

If $Y$ is not affected by $X$ once we know $Z$:
$$
P(Y \mid \text{do}(X), Z) = P(Y \mid Z)
$$

#### **Rule 2: Action/Observation Exchange**

If $Y$ is independent of $X$ given $Z$ and the intervention $\text{do}(X)$:
$$
P(Y \mid \text{do}(X), Z) = P(Y \mid X, Z)
$$

#### **Rule 3: Insertion/Deletion of Actions**

If $Y$ is independent of $X$ given $Z$, then:
$$
P(Y \mid \text{do}(X), Z) = P(Y \mid \text{do}(X), \text{do}(Z))
$$

### **Causal Effect Estimation**

The causal effect of $X$ on $Y$ can be expressed as:
$$
P(Y \mid \text{do}(X=x)) = \sum_z P(Y \mid X=x, Z=z) P(Z=z)
$$

This formula allows us to compute the causal effect by integrating over all possible values of $Z$.

## 4. **Causal Discovery Algorithms**

### **Constraint-Based Methods**

These methods use conditional independence tests to build the causal graph. **Begins with a fully connected graph and iteratively removes edges based on conditional independence tests.**

- **PC Algorithm**: Constructs the DAG by testing for conditional independencies.
- **Fas Algorithm**: Similar to the PC algorithm but starts with a fully connected graph and removes edges iteratively.

### **Score-Based Methods**

These methods assign a score to each possible graph and select the one with the best score. **Starts with an empty graph and adds edges to maximize a scoring function, then prunes the graph to refine the causal structure.**

- **GES (Greedy Equivalence Search)**: Searches over equivalence classes of graphs using a scoring function like the Bayesian Information Criterion (BIC). 

### **Hybrid Methods**

Combine aspects of constraint-based and score-based methods for efficiency and accuracy.

- **Fas-GES**: Uses constraint-based methods to reduce the search space for score-based methods.

## **Conclusion**

Causal inference is a powerful tool for understanding and modeling cause-and-effect relationships in data. It combines statistical techniques with domain knowledge to build models that can predict the effects of interventions and uncover the underlying mechanisms driving observed phenomena.

### **Next Steps**

- Explore specific causal discovery algorithms like PC and GES.
- Apply causal inference techniques to a real-world dataset.
- Consider studying advanced topics like causal effect estimation and counterfactual reasoning.

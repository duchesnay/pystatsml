## Negative Log-Likelihood (NLL) for Binary Classification with Sigmoid Activation

### Negative Log-Likelihood (NLL)

**Setup**

- Inputs: $\{(x_i, y_i)\}_{i=1}^n$, with $y_i \in \{0, 1\}$
- Model:  
$$
\hat{p}_i = \sigma(\mathbf{w}^\top \mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}_i}}
$$
- Objective:  
$$
\mathcal{L}_{\text{NLL}} = - \sum_{i=1}^n \log P(y_i \mid \mathbf{x}_i; \mathbf{w})
$$

Since $y_i \in \{0, 1\}$, we model the likelihood as:

$$
P(y_i \mid \mathbf{x}_i; \mathbf{w}) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}
$$

**Step-by-step Expansion**

$$
\mathcal{L}_{\text{NLL}} = - \sum_{i=1}^n \log \left( \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i} \right)
$$

Apply log properties:

$$
= - \sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

Now substitute $\hat{p}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$ where $z_i = \mathbf{w}^\top \mathbf{x}_i$:

- Use:
  $$
  \log(\sigma(z)) = -\log(1 + e^{-z}), \quad \log(1 - \sigma(z)) = -z - \log(1 + e^{-z})
  $$

So the per-example loss becomes:

$$
\ell_i(\mathbf{w}) = - \left[ y_i \log \sigma(z_i) + (1 - y_i) \log (1 - \sigma(z_i)) \right]
$$

$$
= - \left[ y_i (-\log(1 + e^{-z_i})) + (1 - y_i)(-z_i - \log(1 + e^{-z_i})) \right]
$$

Simplify:

$$
\ell_i(\mathbf{w}) = \log(1 + e^{-z_i}) + (1 - y_i) z_i
$$

Therefore, the total loss over $n$ examples is:


**Final Simplified Expression**

$$
\mathcal{L}_{\text{NLL}}(\mathbf{w}) = \sum_{i=1}^n \left[ \log(1 + e^{-z_i}) + (1 - y_i) z_i \right] \quad \text{with } z_i = \mathbf{w}^\top \mathbf{x}_i
$$

Or, equivalently:

$$
\mathcal{L}_{\text{NLL}}(\mathbf{w}) = \sum_{i=1}^n \log\left(1 + e^{-y_i \cdot \mathbf{w}^\top \mathbf{x}_i} \right)
$$

This final form is particularly elegant and often used in optimization routines.


Let me know if you'd like gradient derivation next!



### Gradient of Negative Log-Likelihood (NLL)


**Recap: The Model and Loss**

We have:

- Input–label pairs: $\{(x_i, y_i)\}_{i=1}^n$, where $y_i \in \{0, 1\}$
- Linear logit: $z_i = \mathbf{w}^\top \mathbf{x}_i$
- Sigmoid output:  
  $$
  \hat{p}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
  $$
- NLL loss:
  $$
  \mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
  $$

We aim to compute the gradient $\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w})$


**Step 1: Loss per Sample**

Define per-sample loss:

$$
\ell_i(\mathbf{w}) = -\left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

Take derivative w.r.t. $\mathbf{w}$. Using the chain rule:

$$
\nabla_{\mathbf{w}} \ell_i = \frac{d\ell_i}{d\hat{p}_i} \cdot \frac{d\hat{p}_i}{dz_i} \cdot \frac{dz_i}{d\mathbf{w}}
$$


**Step 2: Compute Gradients**

- Derivative of the loss w.r.t. $\hat{p}_i$:
  $$
  \frac{d\ell_i}{d\hat{p}_i} = -\left( \frac{y_i}{\hat{p}_i} - \frac{1 - y_i}{1 - \hat{p}_i} \right)
  $$

- Derivative of sigmoid:
  $$
  \frac{d\hat{p}_i}{dz_i} = \hat{p}_i(1 - \hat{p}_i)
  $$

- Derivative of $z_i = \mathbf{w}^\top \mathbf{x}_i$:
  $$
  \frac{dz_i}{d\mathbf{w}} = \mathbf{x}_i
  $$

Putting it all together:

$$
\nabla_{\mathbf{w}} \ell_i = \left[ \hat{p}_i - y_i \right] \mathbf{x}_i
$$


**Step 3: Final Gradient over Dataset**

Sum over all samples:

$$
\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \sum_{i=1}^n (\hat{p}_i - y_i) \mathbf{x}_i
$$

Or in matrix form, if $\mathbf{X} \in \mathbb{R}^{n \times p}$, $\hat{\mathbf{p}} \in \mathbb{R}^n$, and $\mathbf{y} \in \mathbb{R}^n$:

$$
\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \mathbf{X}^\top (\hat{\mathbf{p}} - \mathbf{y})
$$


**Summary**

- Gradient of binary NLL with sigmoid:
  $$
  \boxed{\nabla_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^n (\sigma(\mathbf{w}^\top \mathbf{x}_i) - y_i)\mathbf{x}_i}
  $$
- In matrix form:  
  $$
  \boxed{\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{X}^\top (\hat{\mathbf{p}} - \mathbf{y})}
  $$

This form is used in logistic regression and binary classifiers trained via gradient descent.



### Hessian matrix (i.e., the matrix of second derivatives) for Negative Log-Likelihood (NLL)


**Recap: Setup**

We are given:
- Dataset: $\{(x_i, y_i)\}_{i=1}^n$, with $y_i \in \{0, 1\}$
- Model:  
  $$
  \hat{p}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}, \quad \text{where } z_i = \mathbf{w}^\top \mathbf{x}_i
  $$
- Loss function:
  $$
  \mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
  $$

We already know the gradient is:

$$
\nabla_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^n (\hat{p}_i - y_i) \mathbf{x}_i
$$


**Goal: Hessian $\nabla^2_{\mathbf{w}} \mathcal{L}$**

We now compute the second derivative of $\mathcal{L}$, i.e., the **Hessian matrix** $\mathbf{H} \in \mathbb{R}^{p \times p}$, where each entry is:

$$
\mathbf{H}_{jk} = \frac{\partial^2 \mathcal{L}}{\partial w_j \partial w_k}
$$

**Step-by-Step Derivation**

Recall:
$$
\nabla_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^n (\hat{p}_i - y_i) \mathbf{x}_i
$$

But note that $\hat{p}_i = \sigma(z_i) = \sigma(\mathbf{w}^\top \mathbf{x}_i)$, so $\hat{p}_i$ depends on $\mathbf{w}$ too.

We differentiate the gradient:

$$
\nabla^2_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^n \nabla_{\mathbf{w}} \left[ (\hat{p}_i - y_i) \mathbf{x}_i \right]
$$

The only term depending on $\mathbf{w}$ is $\hat{p}_i$. We apply the chain rule:

$$
\nabla_{\mathbf{w}} \hat{p}_i = \sigma'(z_i) \cdot \nabla_{\mathbf{w}} z_i = \hat{p}_i (1 - \hat{p}_i) \mathbf{x}_i
$$

So the outer derivative becomes:

$$
\nabla_{\mathbf{w}} \left[ (\hat{p}_i - y_i) \mathbf{x}_i \right] = \hat{p}_i (1 - \hat{p}_i) \mathbf{x}_i \mathbf{x}_i^\top
$$

Hence:

$$
\boxed{
\nabla^2_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^n \hat{p}_i (1 - \hat{p}_i) \mathbf{x}_i \mathbf{x}_i^\top
}
$$

This is a **weighted sum of outer products** of input vectors.


**Matrix Form**

Let:
- $\mathbf{X} \in \mathbb{R}^{n \times p}$: input matrix (rows = $x_i^\top$)
- $\hat{\mathbf{p}} \in \mathbb{R}^n$: predicted probabilities
- Define $\mathbf{S} = \text{diag}(\hat{p}_i (1 - \hat{p}_i)) \in \mathbb{R}^{n \times n}$

Then the Hessian is:

$$
\boxed{
\nabla^2_{\mathbf{w}} \mathcal{L} = \mathbf{X}^\top \mathbf{S} \mathbf{X}
}
$$


**Summary**

- The Hessian of the NLL loss with sigmoid output is:
  $$
  \nabla^2_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^n \hat{p}_i (1 - \hat{p}_i) \mathbf{x}_i \mathbf{x}_i^\top
  $$
- In matrix form:
  $$
  \nabla^2_{\mathbf{w}} \mathcal{L} = \mathbf{X}^\top \mathbf{S} \mathbf{X}
  $$
- This is **positive semi-definite**, hence the NLL is convex for logistic regression.



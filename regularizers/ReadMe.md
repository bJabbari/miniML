### Overfitting and Underfitting

**Overfitting:** Occurs when a machine learning model learns the training data too well, capturing noise and irrelevant patterns, leading to poor generalization on unseen data.

**Underfitting:** Occurs when a machine learning model is too simple to capture the underlying structure of the data, resulting in poor performance on both the training and test datasets.

### Weight Regularization

Weight regularization is a technique used to address overfitting by adding a penalty term to the loss function, encouraging the model to learn simpler patterns and avoid overly complex solutions.

The overall equation for the total loss $\mathcal{L_{\text{total}}}$ is given by:

$$\mathcal{L_{\text{total}} = L_{\text{original}} + L_{\text{regularization}}}$$

where:
- $\mathcal{L_{\text{original}}}$ is the original loss function (such as cross-entropy loss or mean squared error), representing the discrepancy between the model 
predictions and the ground truth labels.
- $\mathcal{L_{\text{regularization}}}$ is the regularization term, which penalizes the complexity of the model's parameters.

#### L1 Regularization (Lasso)

In L1 regularization, also known as Lasso regularization, the regularization term is the sum of the absolute values of 
the weights. It encourages sparsity in the weight matrix by penalizing the sum of the absolute values of the weights.
 
$$\mathcal{L_{\text{L1}} = \lambda \sum_{i=1}^{n} |w_i|}$$

**Weight Update Rule:**
$$\mathcal{w_i^\text{new} := w_i^\text{old} - \alpha \left( \frac{\partial L_{\text{original}}}{\partial w_i} + \lambda  \text{sign}(w_i) \right)}$$
where:
- $\alpha$ is the learning rate.
- $\lambda$ is the regularization parameter.

#### L2 Regularization (Ridge)

In L2 regularization, also known as Ridge regularization, the regularization term is the sum of the squares of the weights.

$$ \mathcal{L_{\text{L2}} = \lambda \sum_{i=1}^{n} w_i^2}$$

**Weight Update Rule:**
$$\mathcal{w_i^\text{new} := w_i^\text{old} - \alpha \left( \frac{\partial L_{\text{original}}}{\partial w_i} +2 \lambda w_i \right)}$$

#### L1L2 Regularization

Combines L1 and L2 regularization to leverage the benefits of both techniques.
 
$$ \mathcal{L_{\text{L1L2}} = \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2}$$

:bulb: These regularization techniques help prevent overfitting by controlling the complexity of the model, leading to better generalization performance on unseen data.

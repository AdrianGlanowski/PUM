# Math
## Norms
#### Norm L1 (Lasso)
Norm L1 is a sum of absolute differences.
$$ \left \lVert x \right \rVert _1  = \sum{ \lVert x_i \rVert} $$

Useful when key features should be identified and noise need to be ignored, it's likely that only a few features are important.


#### Norm L2 (Ridge)
Norm L2 is a square root of sum of squared differences.
$$ \left \lVert x \right \rVert _2  = \sqrt{\sum{ x_i^2 }} $$

Useful for general purpose regression to prevent over-fitting, it's likely that all features contribute.

#### Frobenious norm
Measures "size" of a matrix as a square root of sum of the absolute squares of its elements (absolute in regards mainly to $\mathbb{C}$ numbers, for $\mathbb{R}$ it doesn't matter)
$$ \left \lVert X \right \rVert _F  = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}{|x_i^2|}} $$

## Convex function
A function that has a shape of "U", meaning that for every two points on the graph, line connecting the dots will be above the function value.
Analytically it means that $f''(x) \geq 0$ over all domain (if twice differentiable).

Formal definition:
$$ \forall_{x_1, x_2 \in \mathbb{D}} \forall_{\lambda \in [0, 1]} f(\lambda x_1​ + (1−\lambda)x_2​) \leq \lambda f(x_1​)+(1−\lambda)f(x_2​) $$

If differentiable (tangent line is always below the graph):
$$ \forall_{x, y \in \mathbb{D}} f(y) \geq f(x)+f'(x)(y−x) $$

If twice differentiable:
$$ \forall_{x \in \mathbb{D}} f''(x) \geq 0 $$

## Vectors and matrices
#### Orthogonal vectors
Vectors $u$, $v$ are orthogonal if their dot product $u \cdot v = 0$

That also means they are perpendicular

#### Orthonormal vectors
Vectors $u$, $v$ are orthonormal if they are orthogonal and both of length $$ \left \lVert u \right \rVert = 1, \left \lVert v \right \rVert = 1$$

#### Unitary matrix
Matrix $A$ is unitary if
$$ A \cdot A^T = A^T \cdot A = I $$

#### Orthogonal matrix
Matrix $A$ is orthogonal (orthonormal) if rows and column are a set of orthogonal (orthonormal) vectors.

properties:
- $ A^{-1} = A^T$
- $ det(A) \in \{-1, 1\}$
- product of two orthogonal matrices is an orthogonal matrix

#### Diagonal matrix
Matrix where all elements except main diagonal is 0
$$ \begin{bmatrix}
a_{11} & 0  & 0 & \dots & 0\\
0 & a_{22}  & 0 & \dots & 0\\
0 & 0  & a_{33} & \dots & 0\\
\vdots & \vdots  & \vdots & \ddots & \vdots\\
0 & 0  & 0 & \dots & a_{nn}\\
\end{bmatrix} $$

#### Hessian
Square matrix of second-order partial derivatives of a scalar-valued function, or scalar field. It describes the local curvature of a function of many variables.$$ \begin{bmatrix}
\frac{\partial^2f}{\partial x_1^2} & \frac{\partial^2f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^2f}{\partial x_1 \partial x_n}  \\
\frac{\partial^2f}{\partial x_2 \partial x_1} & \frac{\partial^2f}{\partial x_2^2} & \dots & \frac{\partial^2f}{\partial x_2 \partial x_n}  \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2f}{\partial x_n \partial x_1} & \frac{\partial^2f}{\partial x_n \partial x_2} & \dots& \frac{\partial^2f}{\partial x_n^2}  \\
\end{bmatrix} $$

Reminder: ${\frac{\partial^2f}{\partial x_i \partial x_j}} = \frac{\partial}{\partial x_i}(\frac{\partial f}{\partial x_j})$

# Numeric methods
## Matrix factorization
Representing a matrix as a products of other matrices
#### SVD (Singular Value Decomposition)
For any matrix $X \in M_{n,m}(\mathbb R)$ there always exist a unique representation
$$ X = U \Sigma V^T =
\begin{bmatrix}
| & |  &  & | \\ 
u_1 & u_2 & \dots & u_n \\
| & |  &  & | \\
\end{bmatrix}_{n \times n} 
\begin{bmatrix}
\sigma_1 & 0 & \dots & 0\\ 
0 & \sigma_2 & \dots & 0 \\
\vdots & \vdots  & \ddots & \vdots \\
0 & 0 & \dots & \sigma_m \\
\vdots & \vdots  & \vdots & \vdots\\
0&0&0&0\\
\end{bmatrix}_{n \times m}
\begin{bmatrix}
| & |  &  & | \\ 
v_1 & v_2 & \dots & v_m \\
| & |  &  & | \\
\end{bmatrix}_{m \times m}^T =
\hat U \hat \Sigma V^T
$$

- $U \in M_{n,n}$ - left singular vectors - unitary, columns are vectors orthonormal creating a basis for subspace, eigencolumns, sorted in decreasing importance, somehow represent original columns ($\hat U \in M_{n,m}$ represents first m columns)
- $\Sigma \in M_{n,m}$ - singular values - diagonal, non negative elements, sorted in non-ascending order ($\hat \Sigma \in M_{m,m}$ represents first m values)
- $V^T \in M_{m,m}$ - right singular vectors - unitary, $i$-th column of $V^T$ tells what mixture of $u_j$ to add up to achieve $x_i$
- $u_i, \sigma_i, v_i$ is more important than $u_{i+1}, \sigma_{i+1}, v_{i+1}$ in regards to describing original matrix, thus we can approximate original matrix, by reducing dimensions of SVD (chopping of last vectors and values)
- $\hat U \hat \Sigma V^T$ is called economy SVD and is better as it takes up less memory (especially for $n \gg m$) and represents **exactly** the same matrix.
- $A = \sum_{i=1}^{m} u_i \sigma_i  v_i^T \approx \sum_{i=1}^{k<m} u_i \sigma_i  v_i^T = \tilde U \tilde \Sigma \tilde V^T $ (sum of rank 1 matrices)
- **$ \tilde U $ is no longer unitary**, $ \tilde U^T \tilde U = I_{k,k} $, but $ \tilde U \tilde U^T \neq I$


#### Newton–Raphson method of minimizing
An iterative method for finding the roots of a differentiable function $f(x) = 0$. However, to optimize a twice-differentiable $f$, our goal is to find the roots of $f'$. We can therefore use Newton's method on its derivative $f'$ to find solutions to $f'(x)=0$, also known as the critical points of $f$. This is relevant in optimization, which aims to find (global) minima of the function $f$ given inital $x_0$.

$$x_{i+1} = x_i + t = x_i - \frac{f'(x_i)}{f''(x_i)} $$.

#### Secant method 
An iterative numerical method for finding a zero of a function $f$. Given two initial values $x_0$ and $x_1$, the method proceeds according to the recurrence relation. The derivative is replaced by an approximation, thus it is a quasi-Newton method.
$$x_{i+1} = x_i - f(x_i)\frac{x_i - x_{i-1}}{f(x_i) - f(x_{i-1})}$$

## Linear regression

A model that estimates the relationship between a scalar response (dependent variable) and one or more explanatory variables (regressor or independent variable).
$$ y = X\beta + \epsilon $$
where:
- $y$ - vector of scalar response
- $X$ - matrix of row-vectors containing values of explanatory variables
- $\beta$ - vector of weights regarding explanatory variables
- $\epsilon$ - vector of noise/bias/intercept

Cost function is a function that measures how well model prediction matches actual data, in linear regression is typically MSE (Mean Squares Error)
$$ MSE = \frac{1}{n} \sum_{i=1}^n(Y_i - \hat Y_i)^2 $$

where:
- $n$ - number of data points
- $Y_i$ - actual values 
- $\hat Y_i$ - predicted values

weights are corrected with gradient descent

## ML basics
- **standarization** - preprocessing technique that scales features to have a mean of 0 and standard deviation of 1 (Z-score normalization).
- **min-max scaling** - preprocessing data to fit a certain range eg. [-1,1]
- **bias/variance** - to reduce High bias (Underfitting): Increase model complexity, add features, reduce regularization, or use more advanced algorithms, to reduce high variance (Overfitting): Use more training data, reduce model complexity (e.g., fewer features), apply regularization (L1/L2), or use ensemble methods. 
- **overfitting** - model learn the training data too well and reaches great result on it, but doesn't predict test data well.
- **regularisation** - introducing punishment for learning too complex dependencies, good way to reduce overfitting
- **MSE/MAE** - Mean Squared Error/Mean Absolute Error
- **cross-validation** - spliting data set into baches, and iterating over them choosing one as a validating set and others as training cat
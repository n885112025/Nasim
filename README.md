# Nasim
Deep Search

## Linear regression fit exercise
The slide asks whether a simple linear regression is a good fit for the dataset:

```
x = [-3, -2, -1, 0, 1, 2, 3]
y = [9, 4, 1, 0, 1, 4, 9]
```

These points follow a quadratic pattern (roughly \(y = x^2\)). Fitting a straight line to this symmetric set produces a slope of **0.0**, an intercept of **4.0**, and an \(R^2\) of **0.0**—a flat line through \(y = 4\). That line misses the curvature entirely, so the model **underfits** and is **not a good fit**. A polynomial (e.g., quadratic) regression or another nonlinear model is needed to capture the relationship.

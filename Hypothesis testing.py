import numpy as np
from scipy.stats import t
'''
def z_test(μ, n, x̄, σ, α, tail="two"):
    """
    Parameters:
        μ (float): Population mean (null hypothesis mean).
        n (int): Sample size.
        x̄ (float): Sample mean.
        σ (float): Population standard deviation.
        α (float): Significance level (e.g., 0.05 for 5%).
        tail (str): "one" for one-tailed test, "two" for two-tailed test.

    Returns:
        str: Explanation of the decision
    """

    # Compute Z-score
    z_score = (x̄ - μ) / (σ / np.sqrt(n))

    # Get critical Z-value based on tail type
    if tail == "two":
        z_critical = norm.ppf(1 - α / 2)  # Two-tailed test critical value
        reject = abs(z_score) > z_critical  # Decision rule
        explanation = (
            f"Since |Z-computed| = |{z_score:.2f}| is {'greater' if reject else 'less'} than "
            f"Z-critical = {z_critical:.2f}, we {'reject' if reject else 'do not reject'} H₀."
        )

    elif tail == "one":
        z_critical = norm.ppf(1 - α)  # One-tailed test critical value
        reject = z_score > z_critical or z_score < -z_critical  # Decision rule

        if z_score > 0:  # Right-tailed test
            explanation = (
                f"Since Z-computed = {z_score:.2f} is {'greater' if reject else 'less'} than "
                f"Z-critical = {z_critical:.2f}, we {'reject' if reject else 'do not reject'} H₀."
            )
        else:  # Left-tailed test
            explanation = (
                f"Since Z-computed = {z_score:.2f} is {'less' if reject else 'greater'} than "
                f"-Z-critical = {-z_critical:.2f}, we {'reject' if reject else 'do not reject'} H₀."
            )

    return explanation

# Example usage
print(z_test(23, 42, 23.8, 2.4, 0.02, tail="two"))
print(z_test(23, 42, 22.5, 2.4, 0.05, tail="one"))
'''
#Only making t test since its more practical, z only works for big samples
def t_test(μ, n, x̄, s, α, tail="two"):
    """
    Parameters:
        μ (float): Population mean (null hypothesis mean).
        n (int): Sample size.
        x̄ (float): Sample mean.
        s (float): Sample standard deviation.
        α (float): Significance level (e.g., 0.05 for 5%).
        tail (str): "one" for one-tailed test, "two" for two-tailed test.

    Returns:
        str: Explanation of the decision
    """

    df = n - 1  # Degrees of freedom
    t_score = (x̄ - μ) / (s / np.sqrt(n))  # Compute t-statistic

    if tail == "two":
        t_critical = t.ppf(1 - α / 2, df)  # Two-tailed test critical value
        reject = abs(t_score) > t_critical  # Decision rule
        explanation = (
            f"Since |T-computed| = |{t_score:.2f}| is {'greater' if reject else 'less'} than "
            f"T-critical = {t_critical:.2f}, we {'reject' if reject else 'do not reject'} H₀."
        )

    elif tail == "one":
        t_critical = t.ppf(1 - α, df)  # One-tailed test critical value
        reject = t_score > t_critical or t_score < -t_critical  # Decision rule

        if t_score > 0:  # Right-tailed test
            explanation = (
                f"Since T-computed = {t_score:.2f} is {'greater' if reject else 'less'} than "
                f"T-critical = {t_critical:.2f}, we {'reject' if reject else 'do not reject'} H₀."
            )
        else:  # Left-tailed test
            explanation = (
                f"Since T-computed = {t_score:.2f} is {'less' if reject else 'greater'} than "
                f"-T-critical = {-t_critical:.2f}, we {'reject' if reject else 'do not reject'} H₀."
            )

    return explanation

# Example usage
t_test(23, 42, 23.8, 2.4, 0.02, tail="two")
t_test(23, 42, 22.5, 2.4, 0.05, tail="one")

t_test(69.21, 30, 68.43, 3.72, 0.05)
t_test(65, 16, 68.2, 3.8, 0.05)
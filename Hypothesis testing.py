import numpy as np
from scipy.stats import t
from scipy.stats import f
from scipy.stats import chi2

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


def f_test(sample1, sample2, alpha=0.05):
    """
    Perform an F-test to compare the variances of two independent samples.

    Parameters:
        sample1 (array-like): First sample data.
        sample2 (array-like): Second sample data.
        alpha (float): Significance level (default is 0.05 for 95% confidence).

    Returns:
        str: Interpretation of the test results.
    """

    # Sample sizes
    n1, n2 = len(sample1), len(sample2)

    # Sample variances
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)  # Unbiased sample variance (ddof=1)

    # Ensure larger variance is in numerator
    if var1 > var2:
        F_stat = var1 / var2
        df1, df2 = n1 - 1, n2 - 1
    else:
        F_stat = var2 / var1
        df1, df2 = n2 - 1, n1 - 1

    # Critical values
    F_critical_low = f.ppf(alpha / 2, df1, df2)
    F_critical_high = f.ppf(1 - alpha / 2, df1, df2)

    # P-value
    p_value = 2 * min(f.cdf(F_stat, df1, df2), 1 - f.cdf(F_stat, df1, df2))

    # Interpretation
    print(f"F-statistic: {F_stat:.4f}")
    print(f"Degrees of freedom: {df1}, {df2}")
    print(f"Critical values: {F_critical_low:.4f}, {F_critical_high:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        return "Reject H₀: The variances are significantly different."
    else:
        return "Fail to reject H₀: No significant difference in variances."

# Example usage
sample1 = [10, 12, 9, 11, 13, 14, 15, 8, 10, 11]
sample2 = [8, 9, 7, 6, 10, 8, 9, 5, 7, 6]

result = f_test(sample1, sample2)
print(result)



def chi_square_variance_test(sample, pop_var, alpha=0.05):
    """
    Perform a Chi-Square test for a single variance.

    Parameters:
        sample (array-like): Sample data.
        pop_var (float): Population variance (hypothesized variance).
        alpha (float): Significance level (default is 0.05).

    Returns:
        str: Interpretation of the test results.
    """

    # Sample size and variance
    n = len(sample)
    sample_var = np.var(sample, ddof=1)  # Sample variance (unbiased)

    # Compute Chi-Square statistic
    chi_stat = (n - 1) * sample_var / pop_var

    # Critical values
    chi_critical_low = chi2.ppf(alpha / 2, df=n-1)
    chi_critical_high = chi2.ppf(1 - alpha / 2, df=n-1)

    # P-value
    p_value = 2 * min(chi2.cdf(chi_stat, df=n-1), 1 - chi2.cdf(chi_stat, df=n-1))

    # Interpretation
    print(f"Chi-Square Statistic: {chi_stat:.4f}")
    print(f"Degrees of freedom: {n-1}")
    print(f"Critical values: {chi_critical_low:.4f}, {chi_critical_high:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        return "Reject H₀: The variance is significantly different from the population variance."
    else:
        return "Fail to reject H₀: No significant difference in variance."

# Example usage
sample_data = [12, 15, 14, 10, 9, 16, 12, 14, 13, 15]
hypothesized_variance = 4  # Suppose we expect the population variance to be 4

result = chi_square_variance_test(sample_data, hypothesized_variance)
print(result)

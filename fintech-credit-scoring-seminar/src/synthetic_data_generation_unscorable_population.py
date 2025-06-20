#!/home/frederickerleigh/Dokumente/Fintech\ Seminar/FinalCode/FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/venv/bin/python
# %% [markdown]
# # Synthetic Digital Footprint Data Generation
# 
# This notebook generates synthetic digital footprint variables for credit scoring analysis, using published distributions and correlations from Berg et al. (2020). # Synthetic Digital Footprint Data with Copula and Cramer's V Dependency Structure
# Generate a synthetic dataset using marginal distributions and the Cramer's V matrix [shrunken to 14 variables] from Berg et al. (2020).
# 
# 

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm


# %% [markdown]
# ## Variables and Definitions
# 
# ### Digital Footprint Variables (from Table 2)
# - **Credit bureau score (quintile):** [1-5], quintiles of external credit score.
# - **Device type:** Desktop, Tablet, Mobile.
# - **Operating system:** Windows, iOS, Android, Macintosh, Other.
# - **E-mail host:** Gmx, Web, T-Online, Gmail, Yahoo, Hotmail, Other.
# - **Channel:** Paid, Direct, Affiliate, Organic, Other.
# - **Checkout time:** Evening (6pm-midnight), Night (midnight-6am), Morning (6am-noon), Afternoon (noon-6pm).
# - **Do-not-track setting:** Yes/No.
# - **Name in e-mail:** Yes/No (real name present in e-mail).
# - **Number in e-mail:** Yes/No (number present in e-mail).
# - **Is lowercase:** Yes/No (e-mail is all lowercase).
# - **E-mail error:** Yes/No (typo in e-mail).
# 
# ### Control Variables (see Table A1, regression notes)
# - **Age:** In years, from credit bureau (simulate as int 18-80 if no marginal).
# - **Gender:** Female/Male (simulate as Bernoulli, check marginal in paper).
# - **Order amount:** Purchase amount in EUR (simulate with log-normal or normal, see paper for mean/stdev if available).
# - **Item category:** 16 categories (simulate as categorical, uniform if no marginal).
# - **Month:** Categorical (Oct 2015-Dec 2016, i.e., 15 categories).
# 
# If more marginals are available in the paper, add them here.
# 

# %% [markdown]
# ##Define Variable Schemas (Frequencies & Default Rates)
# 

# %%
N = 100000  # Number of synthetic records for strong tail coverage
schemas = {
    "credit_score_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "device_type": ["Desktop", "Tablet", "Mobile", "Do-not-track"],
    "os": ["Windows", "iOS", "Android", "Macintosh", "Other", "Do-not-track"],
    "email_host": ["Gmx", "Web", "T-Online", "Gmail", "Yahoo", "Hotmail", "Other"],
    "channel": ["Paid", "Direct", "Affiliate", "Organic", "Other", "Do-not-track"],
    "checkout_time": ["Evening", "Night", "Morning", "Afternoon"],
    "do_not_track_setting": ["No", "Yes"],
    "name_in_email": ["No", "Yes"],
    "number_in_email": ["No", "Yes"],
    "is_lowercase": ["No", "Yes"],
    "email_error": ["No", "Yes"],
    # ... add controls below ...
    "age_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],  # or "age" if you use actual age buckets
    "gender": ["Female", "Male"],
    "order_amount": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "item_category": ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"],  # fill with actual categories if possible
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
}


# Set marginal frequencies as you did before (proportions must sum to 1 for each variable)

marginals = {
    "credit_bureau_quintile":     [0.20, 0.20, 0.20, 0.20, 0.20],  # Even quintiles
    "device_type":                [0.59, 0.17, 0.10, 0.14],  # Desktop, Tablet, Mobile, Do-not-track (from Table 2)
    "operating_system":           [0.50, 0.16, 0.11, 0.09, 0.01, 0.14], # Windows, iOS, Android, Macintosh, Other, Do-not-track
    "email_host":                 [0.24, 0.21, 0.11, 0.11, 0.05, 0.04, 0.25], # gmx, web, t-online, gmail, yahoo, hotmail, other
    "channel":                    [0.41, 0.21, 0.9, 0.08, 0.07, 0.14],  # Paid, Direct, Affiliate, Organic, Other, Do-not-track
    "checkout_time":              [0.41, 0.02, 0.19, 0.38],  # Evening, Night, Morning, Afternoon
    "do_not_track":               [0.86, 0.14],  # No, Yes
    "name_in_email":              [0.28, 0.72],  # No, Yes (from Table 2)
    "number_in_email":            [0.83, 0.17],  # No, Yes
    "is_lowercase":               [0.93, 0.07],  # No, Yes
    "email_error":                [0.98, 0.02],  # No, Yes
    "age":                        None, # Placeholder; see note below.
    "gender":                     [0.66, 0.34], # If not available, simulate as balanced.
    "order_amount":               None, # Placeholder; see note below.
    "item_category":              [1/16.]*16, # Uniform if nothing better.
    "month":                      [1/15.]*15  # Uniform over 15 months if nothing better.
}
# Note: Age/order amount can be simulated as normal/lognormal if no distribution found - see next cell.


# %% [markdown]
# simultate the age and the order amounts

# %%
import numpy as np
import pandas as pd

N = 100000  # Sample size, consistent with earlier definition

# --- Age: Normal Distribution, Clipped to Empirical Range ---
age_mean = 38.2  # Mean from paper
age_std = 10.46   # Standard deviation from paper
age_min = 18      # Minimum age
age_max = 80      # Maximum age (small % >70, but 80 as upper bound)

ages = np.random.normal(loc=age_mean, scale=age_std, size=N)
ages = np.clip(np.round(ages), age_min, age_max).astype(int)

# --- Order Amount: Log-Normal Distribution, Matched to Mean and Median ---
# Paper reports: mean = 318, median = 219, sd = 317, IQR = 120-400
# For log-normal: median = exp(mu), mean = exp(mu + sigma^2/2)
order_median = 221.6  # Median from paper
order_mean = 324.57    # Mean from paper

mu = np.log(order_median)               # mu = ln(median)
sigma = np.sqrt(2 * (np.log(order_mean) - mu))  # Solve for sigma

order_amounts = np.random.lognormal(mean=mu, sigma=sigma, size=N)
order_amounts = np.clip(order_amounts, 10, 1500)  # Clip to plausible range
order_amounts = np.round(order_amounts, 2)        # Round to 2 decimal places

# Initialize synthetic DataFrame with continuous variables
synthetic = pd.DataFrame({
    "age": ages,
    "order_amount": order_amounts
})

# Bin into quintiles for use with Cramer's V matrix
synthetic['age_quintile'] = pd.qcut(synthetic['age'], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
synthetic['order_amount_quintile'] = pd.qcut(synthetic['order_amount'], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

# Display first few rows to verify
synthetic.head()

# %% [markdown]
# ##Load and Clean the Cramer's V Matrix

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm

# Define variables in the order of the Cramer's V matrix
variables = [
    "credit_score_quintile",
    "device_type",
    "os",
    "email_host",
    "channel",
    "checkout_time",
    "name_in_email",
    "number_in_email",
    "is_lowercase",
    "email_error",
    "age_quintile",
    "order_amount_quintile",
    "item_category",
    "month"
]

# Define categories for each variable (consistent with schemas)
categories = {
    "credit_score_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "device_type": ["Desktop", "Tablet", "Mobile", "Do-not-track"],
    "os": ["Windows", "iOS", "Android", "Macintosh", "Other", "Do-not-track"],
    "email_host": ["Gmx", "Web", "T-Online", "Gmail", "Yahoo", "Hotmail", "Other"],
    "channel": ["Paid", "Direct", "Affiliate", "Organic", "Other", "Do-not-track"],
    "checkout_time": ["Evening", "Night", "Morning", "Afternoon"],
    "name_in_email": ["No", "Yes"],
    "number_in_email": ["No", "Yes"],
    "is_lowercase": ["No", "Yes"],
    "email_error": ["No", "Yes"],
    "age_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "order_amount_quintile": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "item_category": [f"Cat{i}" for i in range(1, 17)],  # 16 categories
    "month": ["Oct15", "Nov15", "Dec15", "Jan16", "Feb16", "Mar16", "Apr16", 
              "May16", "Jun16", "Jul16", "Aug16", "Sep16", "Oct16", "Nov16", "Dec16"]  # 15 months
}

# Define marginal probabilities (from your marginals dictionary, adjusted)
marginals_list = [
    [0.20, 0.20, 0.20, 0.20, 0.20],  # credit_score_quintile
    [0.59, 0.17, 0.10, 0.14],        # device_type
    [0.5, 0.16, 0.11, 0.09, 0.01, 0.14],  # os
    [0.24, 0.21, 0.11, 0.11, 0.05, 0.04, 0.25],  # email_host
    [0.41, 0.21, 0.09, 0.08, 0.07, 0.14],  # channel
    [0.41, 0.02, 0.19, 0.38],        # checkout_time
    [0.28, 0.72],                    # name_in_email
    [0.83, 0.17],                    # number_in_email
    [0.93, 0.07],                    # is_lowercase
    [0.98, 0.02],                    # email_error
    [0.20, 0.20, 0.20, 0.20, 0.20],  # age_quintile
    [0.20, 0.20, 0.20, 0.20, 0.20],  # order_amount_quintile
    [1/16]*16,                       # item_category (uniform over 16)
    [1/15]*15                        # month (uniform over 15)
]

# Load Cramer's V matrix (from your existing code)
cramers_v_array = np.array([
    [1.00, 0.07, 0.05, 0.07, 0.03, 0.03, 0.01, 0.07, 0.02, 0.00, 0.2, 0.01, 0.05, 0.01],
    [0.07, 1.00, 0.71, 0.07, 0.06, 0.04, 0.05, 0.06, 0.07, 0.01, 0.12, 0.03, 0.05, 0.06],
    [0.05, 0.71, 1.00, 0.08, 0.06, 0.04, 0.06, 0.08, 0.06, 0.01, 0.1, 0.02, 0.04, 0.03],
    [0.07, 0.07, 0.08, 1.00, 0.03, 0.03, 0.08, 0.18, 0.04, 0.06, 0.16, 0.02, 0.02, 0.01],
    [0.03, 0.06, 0.06, 0.03, 1.00, 0.02, 0.01, 0.02, 0.04, 0.02, 0.09, 0.04, 0.06, 0.13],
    [0.03, 0.04, 0.04, 0.03, 0.02, 1.00, 0.01, 0.01, 0.01, 0.01, 0.06, 0.01, 0.03, 0.02],
    [0.01, 0.05, 0.06, 0.08, 0.01, 0.01, 1.00, 0.22, 0.01, 0.02, 0.04, 0.01, 0.03, 0.01],
    [0.07, 0.06, 0.08, 0.18, 0.02, 0.01, 0.22, 1.00, 0.02, 0.00, 0.06, 0.01, 0.04, 0.01],
    [0.02, 0.07, 0.06, 0.04, 0.04, 0.01, 0.01, 0.02, 1.00, 0.03, 0.03, 0.02, 0.02, 0.02],
    [0.00, 0.01, 0.01, 0.06, 0.02, 0.01, 0.02, 0.00, 0.03, 1.00, 0.03, 0.01, 0.01, 0.01],
    [0.2, 0.12, 0.1, 0.16, 0.09, 0.06, 0.04, 0.06, 0.03, 0.03, 1.00, 0.05, 0.11, 0.03],
    [0.01, 0.03, 0.02, 0.02, 0.04, 0.01, 0.01, 0.01, 0.02, 0.01, 0.05, 1.00, 0.27, 0.02],
    [0.05, 0.05, 0.04, 0.02, 0.06, 0.03, 0.03, 0.04, 0.02, 0.01, 0.11, 0.27, 1.00, 0.11],
    [0.01, 0.06, 0.03, 0.01, 0.13, 0.02, 0.01, 0.01, 0.02, 0.01, 0.03, 0.02, 0.11, 1.00]
])

# Check if Cramer's V matrix is positive semi-definite
eigvals = np.linalg.eigvalsh(cramers_v_array)
if np.any(eigvals < 0):
    print("Warning: Cramer's V matrix is not positive semi-definite. Adjusting to identity for simplicity.")
    cramers_v_array = np.eye(14)  # Fallback to independent variables

# Compute thresholds for each variable
thresholds_list = []
for p in marginals_list:
    cumprob = np.cumsum(p)[:-1]  # Cumulative probs excluding 1.0
    thresholds = norm.ppf(cumprob)
    thresholds_list.append(thresholds)

# Generate multivariate normal data with Cramer's V as covariance
Z = np.random.multivariate_normal(mean=np.zeros(14), cov=cramers_v_array, size=N)

# Map continuous Z to categorical variables
for i, var in enumerate(variables):
    thresholds = thresholds_list[i]
    z = Z[:, i]
    # Assign categories based on where z falls relative to thresholds
    cat_indices = np.searchsorted(thresholds, z, side='right')
    synthetic[var] = [categories[var][idx] for idx in cat_indices]

# Add gender independently (not in Cramer's V matrix)
synthetic['gender'] = np.random.choice(['Female', 'Male'], size=N, p=[0.66, 0.34])

# Display first few rows to verify
synthetic.head()


# %% [markdown]
# # Inspect the shape and preview the data (without default rates)

# %%
# Preview shape and sample of the synthetic dataset
print("Synthetic data shape:", synthetic.shape)
synthetic.head()
synthetic.describe(include='all')  # Quick summary for all columns


# %% [markdown]
# Assign Default Status (TARGET) Using Per-Category Rates

# %%
# Define the default_rates dictionary (as above, use the categories from your schemas)
default_rates = {
    "credit_score_quintile": {"Q1": 0.0212, "Q2": 0.0102, "Q3": 0.0068, "Q4": 0.0047, "Q5": 0.0039},
    "device_type": {"Desktop": 0.0216, "Tablet": 0.0164, "Mobile": 0.0621, "Do-not-track": 0.0228},
    "os": {"Windows": 0.0219, "iOS": 0.0235, "Android": 0.0480, "Macintosh": 0.0169, "Other": 0.0745, "Do-not-track": 0.0228},
    "email_host": {"Gmx": 0.0242, "Web": 0.0263, "T-Online": 0.0152, "Gmail": 0.0361, "Yahoo": 0.0315, "Hotmail": 0.0275, "Other": 0.0222},
    "channel": {"Paid": 0.0289, "Direct": 0.0187, "Affiliate": 0.0265, "Organic": 0.0255, "Other": 0.0215, "Do-not-track": 0.0228},
    "checkout_time": {"Evening": 0.0205, "Night": 0.0352, "Morning": 0.0274, "Afternoon": 0.0278},
    "name_in_email": {"No": 0.0124, "Yes": 0.0082},
    "number_in_email": {"No": 0.0084, "Yes": 0.0141},
    "is_lowercase": {"No": 0.0084, "Yes": 0.0214},
    "email_error": {"No": 0.0088, "Yes": 0.0509},
}

# Assign the TARGET column
cat_vars = [var for var in default_rates if var in synthetic.columns]
N = synthetic.shape[0]
default_probs = np.zeros(N)
for var in cat_vars:
    default_probs += synthetic[var].map(default_rates[var]).values
default_probs /= len(cat_vars)

# Apply precise calibration to reach exactly 2.5% default rate
TARGET_DEFAULT_RATE = 0.025  # 2.5% default rate for unscorable population
current_mean = default_probs.mean()
scaling_factor = TARGET_DEFAULT_RATE / current_mean
adjusted_probs = np.clip(default_probs * scaling_factor, 0, 1)

# Assign TARGET using adjusted probabilities
# Method 1: Direct application with random sampling
synthetic['TARGET'] = (np.random.rand(N) < adjusted_probs).astype(int)

# Check if we need further adjustment to hit exactly 2.5%
if abs(synthetic['TARGET'].mean() - TARGET_DEFAULT_RATE) > 0.001:
    # Method 2: Sort and select exact number of defaults
    num_defaults_needed = int(N * TARGET_DEFAULT_RATE)
    sorted_indices = np.argsort(adjusted_probs)[::-1]  # Highest default risk first
    synthetic['TARGET'] = 0  # Reset all to non-default
    synthetic.loc[synthetic.index[sorted_indices[:num_defaults_needed]], 'TARGET'] = 1  # Set top N to default

print(f"Target default rate: {TARGET_DEFAULT_RATE:.1%}")
print(f"Achieved default rate: {synthetic['TARGET'].mean():.4%}")
n_defaults = synthetic['TARGET'].sum()
n_total = len(synthetic)
print(f"Number of defaults: {n_defaults} out of {n_total} ({n_defaults / n_total:.4%})")


# %% [markdown]
# Compare Summary Statistics to Berg et al.

# %%
# Summarize categorical variables: frequency tables
for col in cat_vars:
    print(f"\n{col} value counts:")
    print(synthetic[col].value_counts(normalize=True).sort_index())

# For continuous (age, order_amount)
print("\nAge summary:")
print(synthetic['age'].describe())
print("\nOrder amount summary:")
print(synthetic['order_amount'].describe())

# Default rate by category (optional, as in Table 2)
for col in cat_vars:
    print(f"\nDefault rate by {col}:")
    print(synthetic.groupby(col)['TARGET'].mean().round(4))


# %% [markdown]
# Export Your Synthetic Data

# %%
synthetic.to_csv("/home/frederickerleigh/Dokumente/Fintech Seminar/FinalCode/FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/data/synthetic_digital_footprint_with_target_unscorable.csv", index=False)


# %% [markdown]
# ## Copula-Based Synthetic Data Generation
# 
# This section generates synthetic digital footprint data using a Gaussian copula, the empirical marginals, and the Cramer's V matrix from Berg et al. (2020), following standard practices in synthetic data literature.
# 

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm

# Assume N, categories, marginals_list, and cramers_v_array are already defined and match your previous cells.
# The order of variables must match that of cramers_v_array!
variables = [
    "credit_score_quintile", "device_type", "os", "email_host", "channel", "checkout_time",
    "name_in_email", "number_in_email", "is_lowercase", "email_error",
    "age_quintile", "order_amount_quintile", "item_category", "month"
]

N = 100000 # or whatever your sample size is


# %% [markdown]
# ### Compute Thresholds
# 
# For each variable, we convert marginal probabilities into Z-score cut points for the copula simulation.
# 

# %%
# For each variable, compute thresholds that divide the standard normal into the same proportions as marginals
thresholds_list = []
for marg in marginals_list:
    cumprob = np.cumsum(marg)[:-1]
    thresholds = norm.ppf(cumprob)
    thresholds_list.append(thresholds)


# %% [markdown]
# ### Simulate Correlated Normal Latent Variables
# 
# Use the Cramer's V matrix as a dependency structure (covariance).
# 

# %%
# 1. Check Cramer's V matrix (should be positive semi-definite)
eigvals = np.linalg.eigvalsh(cramers_v_array)
if np.any(eigvals < 0):
    print("Warning: Cramer's V matrix not PSD. Using identity (independence) instead.")
    cramers_v_array = np.eye(len(variables))

# 2. Simulate multivariate normal
Z = np.random.multivariate_normal(mean=np.zeros(len(variables)), cov=cramers_v_array, size=N)

# 3. Map each column to category using thresholds
copula_df = pd.DataFrame()
for i, var in enumerate(variables):
    thresh = thresholds_list[i]
    z_col = Z[:, i]
    # Assign categories based on cut points
    idx = np.searchsorted(thresh, z_col, side='right')
    copula_df[var] = [categories[var][j] for j in idx]

# Add continuous variables from synthetic DataFrame
copula_df['age'] = synthetic['age']
copula_df['order_amount'] = synthetic['order_amount']

# %% [markdown]
# ### Add Gender and Continuous Controls
# 
# Add gender independently (if needed), and join your simulated age/order_amount as before.
# 

# %%
copula_df['gender'] = np.random.choice(['Female', 'Male'], size=N, p=[0.66, 0.34])
copula_df['age'] = synthetic['age']
copula_df['order_amount'] = synthetic['order_amount']

# %% [markdown]
# ### Assign Default Status
# 
# Assign TARGET with precise control to match exactly 2.5% default rate
# 

# %%
TARGET_DEFAULT_RATE = 0.025  # 2.5% default rate for unscorable population

cat_vars = [var for var in default_rates if var in copula_df.columns]
default_probs = np.zeros(N)
for var in cat_vars:
    default_probs += copula_df[var].map(default_rates[var]).values
default_probs /= len(cat_vars)

# Apply precise calibration
current_mean = default_probs.mean()
scaling_factor = TARGET_DEFAULT_RATE / current_mean
adjusted_probs = np.clip(default_probs * scaling_factor, 0, 1)

# Method 2: Sort and select exact number of defaults
num_defaults_needed = int(N * TARGET_DEFAULT_RATE)
sorted_indices = np.argsort(adjusted_probs)[::-1]  # Highest default risk first
copula_df['TARGET'] = 0  # Reset all to non-default
copula_df.loc[copula_df.index[sorted_indices[:num_defaults_needed]], 'TARGET'] = 1  # Set top N to default

print(f"Copula synthetic default rate target: {TARGET_DEFAULT_RATE:.1%}")
print(f"Copula synthetic default rate actual: {copula_df['TARGET'].mean():.4%}")


# %% [markdown]
# ### Inspect and Save
# 
# Check the summary and optionally export your copula-generated synthetic dataset.
# 

# %%
copula_df.to_csv("/home/frederickerleigh/Dokumente/Fintech Seminar/FinalCode/FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/data/synthetic_digital_footprint_copula_unscorable.csv", index=False)
print("Copula synthetic data", copula_df.shape)
copula_df.head()


# %% [markdown]
# ## CTGAN-Based Synthetic Data Generation
# 
# In this section, we use the Conditional Tabular GAN (CTGAN) model from the SDV library to generate synthetic digital footprint data. CTGAN learns the distributions and relationships directly from the seed data and produces high-fidelity synthetic samples, using a neural network-based approach.
# 

# %%
# If not installed, run this in a terminal/cell:
# !pip install sdv

from ctgan import CTGAN
import pandas as pd
import numpy as np

# Load your original synthetic dataframe (the "classic" one with all columns, including TARGET)
seed_df = pd.read_csv("/home/frederickerleigh/Dokumente/Fintech Seminar/FinalCode/FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/data/synthetic_digital_footprint_with_target_unscorable.csv")  # or just use 'synthetic' if in memory

# If you need to limit the number of samples, do so here (optional):
# seed_df = seed_df.sample(n=10000, random_state=42)


# %% [markdown]
# ### Train CTGAN
# 
# We train the CTGAN model on the seed data for a modest number of epochs. More epochs = higher fidelity, but longer runtime. For research, 100-300 epochs is common.
# 

# %%
ctgan = CTGAN(epochs=300, verbose=True, cuda=True)  # Use cuda=True if on GPU

# Optional: List categorical columns for CTGAN
categorical_columns = [
    "credit_score_quintile", "device_type", "os", "email_host", "channel", "checkout_time",
    "name_in_email", "number_in_email", "is_lowercase", "email_error",
    "age_quintile", "order_amount_quintile", "item_category", "month", "gender"
]
# You can exclude "TARGET" if treating it as the output, or include for conditional generation.

ctgan.fit(seed_df, discrete_columns=categorical_columns)


# %% [markdown]
# ### Sample Synthetic Data with Exact Default Rate
# 
# Generate synthetic data with controlled default rate, fixing it at exactly 2.5% for the unscorable population.
# 

# %%
TARGET_DEFAULT_RATE = 0.025  # 2.5% default rate for unscorable population
N = len(seed_df)

# Generate synthetic data using CTGAN
ctgan_synth = ctgan.sample(N)

# Apply the same precise calibration approach as used for other methods
# Calculate default probabilities based on feature values
cat_vars = [var for var in default_rates if var in ctgan_synth.columns]
default_probs = np.zeros(len(ctgan_synth))
for var in cat_vars:
    # Handle cases where CTGAN might generate categories not in default_rates
    mapped_values = ctgan_synth[var].map(default_rates[var])
    # Fill NaN values with mean default rate for that variable
    mean_rate = np.mean(list(default_rates[var].values()))
    mapped_values = mapped_values.fillna(mean_rate)
    default_probs += mapped_values.values
default_probs /= len(cat_vars)

# Apply precise calibration
current_mean = default_probs.mean()
scaling_factor = TARGET_DEFAULT_RATE / current_mean
adjusted_probs = np.clip(default_probs * scaling_factor, 0, 1)

# Sort and select exact number of defaults
num_defaults_needed = int(N * TARGET_DEFAULT_RATE)
sorted_indices = np.argsort(adjusted_probs)[::-1]  # Highest default risk first
ctgan_synth['TARGET'] = 0  # Reset all to non-default
ctgan_synth.loc[sorted_indices[:num_defaults_needed], 'TARGET'] = 1  # Set top N to default

# Verify the resulting default rate
print(f"CTGAN synthetic default rate target: {TARGET_DEFAULT_RATE:.1%}")
print(f"CTGAN synthetic default rate actual: {ctgan_synth['TARGET'].mean():.4%}")

# Save the data
ctgan_synth.to_csv("/home/frederickerleigh/Dokumente/Fintech Seminar/FinalCode/FintechSeminar-Synthetic-Dataset/fintech-credit-scoring-seminar/data/synthetic_digital_footprint_ctgan_unscorable.csv", index=False)


# %% [markdown]
# ### Quick Check
# 
# Preview the summary statistics and value counts for key columns.
# 

# %%
print(ctgan_synth.head())
print(ctgan_synth.describe(include='all'))

# Optionally, value counts for categorical columns
for col in categorical_columns:
    print(f"{col} value counts:")
    print(ctgan_synth[col].value_counts(normalize=True).sort_index())




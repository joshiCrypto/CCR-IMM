#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:41:58 2025

@author: joshuakaji
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
n_stocks = 50
n_days = 250
initial_price = 100  # Initial stock price


# volatility/Drift params (Annualized)
drift_market = 0.05
vol_market = 0.8
vol_smb = 0.4
vol_hml = 0.2
vol_noise = 0.3

# Simulate Fama-French Factors
np.random.seed(42)
market_excess_return = np.random.normal(0.0005, vol_market*np.sqrt(dt), n_days)  # Market factor
smb = np.random.normal(0, vol_smb*np.sqrt(dt), n_days)  # SMB factor
hml = np.random.normal(0, vol_hml*np.sqrt(dt), n_days)  # HML factor

# Generate random betas for stocks
betas = {
    "market": np.random.choice([0.5, 1], n_stocks),
    "smb": np.random.choice([-1, 1], n_stocks),
    "hml": np.random.choice([-1, 1], n_stocks),
    "alpha": np.random.choice([0.0], n_stocks)  # Stock-specific alpha
}

# Generate stock returns based on Fama-French Model
returns = np.zeros((n_days, n_stocks))
for i in range(n_stocks):
    noise = np.random.normal(0, vol_noise*np.sqrt(dt), n_days)  # Stock-specific residual noise
    returns[:, i] = (
        betas["alpha"][i]
        + betas["market"][i] * market_excess_return
        + betas["smb"][i] * smb
        + betas["hml"][i] * hml
        + noise
    )

# Convert returns to prices using GBM
prices = np.zeros((n_days, n_stocks))
prices[0, :] = initial_price

for t in range(1, n_days):
    prices[t, :] = prices[t - 1, :] * np.exp(returns[t, :])

# Create a DataFrame for analysis
price_df = pd.DataFrame(prices, columns=[f"Stock_{i+1}" for i in range(n_stocks)])

# Plot some stock prices

price_df.plot(legend=False)

lr_df = np.log(price_df).diff().dropna()

############################################################
############ PCA
X = (lr_df - lr_df.mean()).values
#log_returns_standardized = StandardScaler().fit_transform(lr_df.values)
pca = PCA(n_components=X.shape[1])  # Number of components = number of columns
pca_scores = pca.fit(X)  # Project data onto principal components

# princple components
P = pca.components_.T 

# eigen values 
eigenvalues = pca.explained_variance_
explained_variance = eigenvalues/eigenvalues.sum()
summary_variance = pd.DataFrame(explained_variance)
summary_variance[1] = summary_variance.cumsum()
summary_variance.columns = ["Variance", "cumulated Variance"]

summary_variance.iloc[:10].plot.bar(xlabel="Principle Component", 
                                                 ylabel="variance explained")


############################################################
#### Projection on first 5 Principle components 
fig, axs = plt.subplots(5, 1, figsize=(8, 14))
for i in range(0, 5):
    pc_i = P[:, i]  # ith principal component
    projection = X.dot(pc_i)  # Shape (252,), projection of X onto pc1
    reconstruction = np.outer(projection, pc_i)  # X*PC1
    pd.DataFrame(reconstruction).cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", ax =axs[i], grid=True)



############################################################
#### Projection on first PC : Market factor 
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
i = 0
pc_i = P[:, i]  # ith principal component
projection = X.dot(pc_i)  # Shape (252,), projection of X onto pc1
lr_proj = pd.DataFrame(np.outer(projection, pc_i))  # X*PC1

mask_beta_market_high = betas["market"] == 1
mask_beta_market_low = betas["market"] == 0.5

lr_proj.loc[:, mask_beta_market_high].cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", 
                                                    ax =axs[0], grid=True, color='blue', alpha=0.5)
lr_proj.loc[:, mask_beta_market_low].cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", 
                                                   ax =axs[0], grid=True, color='red', alpha=0.5)

# plot market factor to compare 
pd.Series(market_excess_return).cumsum().plot(title=f"Market Factor", ax =axs[1], grid=True)
fig.show()

############################################################
#### Projection on first PC : SMB 
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
i = 1
pc_i = P[:, i]  # ith principal component
projection = X.dot(pc_i)  # Shape (252,), projection of X onto pc1
lr_proj = pd.DataFrame(np.outer(projection, pc_i))  # X*PC1

# tODO, color the beta component 
beta_smb_long = betas['smb'] == 1 
beta_smb_short = betas['smb'] == -1

lr_proj.loc[:, beta_smb_long].cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", 
                                                    ax =axs[0], grid=True, color='blue', alpha=0.5)
lr_proj.loc[:, beta_smb_short].cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", 
                                                   ax =axs[0], grid=True, color='red', alpha=0.5)

# plot SMB factor to compare 
pd.Series(smb).cumsum().plot(title=f"Market Factor", ax =axs[1])
fig.show()

############################################################
#### Projection on third PC : BMS 
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
i = 2
pc_i = P[:, i]  # ith principal component
projection = X.dot(pc_i)  # Shape (252,), projection of X onto pc1
lr_proj = pd.DataFrame(np.outer(projection, pc_i))  # X*PC1

# tODO, color the beta component 
beta_hml_long = betas['hml'] == 1 
beta_hml_short = betas['hml'] == -1

lr_proj.loc[:, beta_hml_long].cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", 
                                                    ax =axs[0], grid=True, color='blue', alpha=0.5)
lr_proj.loc[:, beta_hml_short].cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", 
                                                   ax =axs[0], grid=True, color='red', alpha=0.5)

# plot HML factor to compare 
pd.Series(hml).cumsum().plot(title=f"Market Factor", ax =axs[1])
fig.show()


############################################################
#### Projection on 4th, 5th, ...  : Expected Noise 
fig, axs = plt.subplots(5, 1, figsize=(8, 10))
for i in range(3, 3+ 5):
    pc_i = P[:, i]  # ith principal component
    projection = X.dot(pc_i)  # Shape (252,), projection of X onto pc1
    reconstruction = np.outer(projection, pc_i)  # X*PC1
    pd.DataFrame(reconstruction).cumsum().plot(legend=False, title=f"projection on {i+1}th EigenVector", ax =axs[i-3])


############################################################
#### Projection on 4th, 5th, ...  : Expected Noise 
from itertools import chain

chain_rng = chain(range(1, 3+1), range(3+1, n_stocks+1, 10))

fig, axs = plt.subplots(len(list(chain_rng)), 2, figsize=(10, 20), gridspec_kw={'width_ratios': [3, 1]})
## A: The loadings matrix (projection of X onto the principal components)
A = np.dot(P.T, lr_df.values.T).T  # Project data onto principal components
counter = 0
for dim in chain(range(1, 3+1), range(3+1, n_stocks+1, 10)):
    # plot returns on increasing space of eigenvalues
    X_reconstructed = np.dot(P[:, :dim], A[:, :dim].T).T
    df_reconst = pd.DataFrame(X_reconstructed).cumsum()
    pd.DataFrame(X_reconstructed).cumsum().plot(legend=False, title=f"projection on fist {dim} EigenVectors", grid=True,
                                                color='grey', alpha=0.5, 
                                                ax = axs[counter, 0])
    axs[counter, 0].set_title(f"projection on fist {dim} EigenVectors")
    
    # plot pie chart with variance contribution 
    v2 = summary_variance.iloc[:dim]
    v2.index = [f"PC_{i}" for i in range(1, dim+1)] 
    v3 = v2[['Variance']].T
    v3["UnExplained"] = 1- v3.sum(1)

    sizes = v3.iloc[0].values  # Getting the values of the first (and only) row
    labels = v3.columns  # Using the column names as labels

    colors = ['grey'] * (len(sizes) - 1) + ['red']  # Red for unexplained variance

    # Plot pie chart
    #axs[counter, 1].figure(figsize=(7, 7))
    axs[counter, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axs[counter, 1].set_title('Variance Explained by Each \nPrincipal Component and Unexplained Variance')
    
    counter+=1


fig.subplots_adjust(wspace=0.3)



middle_threshold = betas["smb"].mean()
np.quantile(betas["smb"], 0.5)



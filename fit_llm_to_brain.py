import numpy as np
import logging
import sys
import time
import joblib
import matplotlib.pyplot as plt
import torch
import cortex
from transformers import AutoTokenizer, AutoModelForCausalLM # Only necessary for feature extraction.
from transformer_lens.components import TransformerBlock

# Repository imports
from ridge_utils.ridge import bootstrap_ridge
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds
from ridge_utils.DataSequence import DataSequence
from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts_opt
from ridge_utils.tokenization_helpers import convert_to_feature_mats_opt

### Some extra helper functions

zscore = lambda v: (v - v.mean(0)) / v.std(0)
zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1, c2: (zs(c1) * zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""

### Ignore irrelevant warnings that muck up the notebook
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Some parameters
NUM_VOX = 95556 # Number of voxels in the subject we plan to use
NUM_TRS = 790 # Number of TRs across 3 test stories
trim_start = 50 # Trim 50 TRs off the start of the story
trim_end = 5 # Trim 5 off the back
ndelays = 4 # We use 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
delays = range(1, ndelays + 1)
from configs import engram_dir, allstories
import os

box_dir = os.path.join(engram_dir, 'huth_box/')
grids = joblib.load(os.path.join(box_dir, "grids_huge.jbl")) # Load TextGrids containing story annotations
trfiles = joblib.load(os.path.join(box_dir, "trfiles_huge.jbl")) # Load TRFiles containing TR information

wordseqs = make_word_ds(grids, trfiles)
for story in wordseqs.keys():
    wordseqs[story].data = [i.strip() for i in wordseqs[story].data]
print("Loaded text data")
# Explore Participant Responses
response_path = os.path.join(box_dir, 'responses', 'full_responses', 'UTS03_responses.jbl')
resp_dict = joblib.load(response_path)
to_pop = [x for x in resp_dict.keys() if 'canplanetearthfeedtenbillionpeoplepart' in x]
for story in to_pop:
    del resp_dict[story]
train_stories = list(resp_dict.keys())#[:-10]
test_stories = list(resp_dict.keys())[-10:]
print("Loaded participant responses")

# Load LLM
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch

from transformer_lens.utils import tokenize_and_concatenate
from huggingface_hub import login
from configs import huggingface_token

login(token=huggingface_token)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
tokenizer = model.tokenizer
def override_to_local_attn(model, window_size=512):
    for b in model.blocks:  # Possibly a cleaner way by correctly using 'use_local_attn'
        if isinstance(b, TransformerBlock):
            n_ctx = b.attn.cfg.n_ctx
            attn_mask = torch.zeros((n_ctx, n_ctx)).bool()
            for i in range(n_ctx):
                start_idx = max(0, i-window_size)
                attn_mask[i, start_idx:i+1] = True
            b.attn.mask = attn_mask.to(device)

override_to_local_attn(model)
def find_word_boundaries(text_data, tokenizer):
    full_story = " ".join(text_data).strip()
    tokenized_story = tokenizer(full_story)['input_ids']

    word_boundaries = []  # In the tokenized story
    curr_word_idx = 0
    curr_word = text_data[curr_word_idx]
    curr_token_set = []

    if curr_word == '':
        curr_word_idx += 1
        curr_word = text_data[curr_word_idx]
        word_boundaries.append(1)

    for token_idx, token in enumerate(tokenized_story):
        curr_token_set.append(token)
        detokenized_chunk = tokenizer.decode(curr_token_set)
        if curr_word in detokenized_chunk:
            word_boundaries.append(token_idx)
            curr_word_idx += 1
            if curr_word_idx == len(text_data):
                break
            curr_word = text_data[curr_word_idx]
            curr_token_set = []

            if curr_word == '':  # Edge case
                word_boundaries.append(token_idx)
                curr_word_idx += 1
                if curr_word_idx == len(text_data):
                    break
                curr_word = text_data[curr_word_idx]

    return tokenized_story, word_boundaries

llm_train_responses = []
for train_story in train_stories:
    ws = wordseqs[train_story]
    text_data = ws.data
    tokenized_story, word_boundaries = find_word_boundaries(text_data, tokenizer)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            torch.tensor(tokenized_story).to(device),
            prepend_bos=True,
            names_filter=lambda name: name.startswith('blocks.7.hook_resid_post'),
        )
    llm_response = cache['blocks.7.hook_resid_post'][0, word_boundaries, :]
    llm_data_seq = DataSequence(llm_response.cpu().numpy(), ws.split_inds, ws.data_times, ws.tr_times)
    interp_llm_response = llm_data_seq.chunksums('lanczos', window=3)
    interp_llm_response = ridge_utils.npp.zs(interp_llm_response[10:-5])
    llm_train_responses.append(interp_llm_response)

llm_train_responses = np.vstack(llm_train_responses)
print(llm_train_responses.shape)
llm_test_responses = []
for test_story in test_stories:
    ws = wordseqs[test_story]
    text_data = ws.data
    tokenized_story, word_boundaries = find_word_boundaries(text_data, tokenizer)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            torch.tensor(tokenized_story).to(device),
            prepend_bos=True,
            names_filter=lambda name: name.startswith('blocks.7.hook_resid_post'),
        )
    llm_response = cache['blocks.7.hook_resid_post'][0, word_boundaries, :]
    llm_data_seq = DataSequence(llm_response.cpu().numpy(), ws.split_inds, ws.data_times, ws.tr_times)
    interp_llm_response = llm_data_seq.chunksums('lanczos', window=3)
    interp_llm_response = ridge_utils.npp.zs(interp_llm_response[10:-5])
    llm_test_responses.append(interp_llm_response[40:])

llm_test_responses = np.vstack(llm_test_responses)
print(llm_test_responses.shape)
del cache
torch.cuda.empty_cache()

# Add FIR delays
delRstim = make_delayed(llm_train_responses, delays)
delPstim = make_delayed(llm_test_responses, delays)

alphas = np.logspace(1, 4, 15) # Equally log-spaced ridge parameters between 10 and 10000. 
nboots = 5 # Number of cross-validation ridge regression runs. You can lower this number to increase speed.

Rresp = np.vstack([resp_dict[story] for story in train_stories])
Presp = np.vstack([resp_dict[story][40:] for story in test_stories])
# Run Regression
np.random.seed(0)
selected_features = np.random.choice(Rresp.shape[1], size=5000, replace=False)
np.random.seed()
X_train = delRstim
X_test = delPstim
Y_train = Rresp[:, selected_features]
Y_test = Presp[:, selected_features]
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Define the range of alphas for cross-validation
alphas = np.logspace(1, 5, 15)  # Regularization strengths

# Create a pipeline with standard scaling and ridge regression
ridge_pipeline = make_pipeline(
    StandardScaler(),  # Standardize features by removing the mean and scaling to unit variance
    Ridge()  # Ridge regression
)

# Set up the parameter grid for alpha
param_grid = {'ridge__alpha': alphas}

# Use GridSearchCV to perform cross-validation with parallel processing
grid_search = GridSearchCV(
    ridge_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=10
)

# Fit the model
grid_search.fit(X_train, Y_train)

# Retrieve the best alpha
best_alpha = grid_search.best_params_['ridge__alpha']
print(f"Best alpha: {best_alpha}")
# After fitting the grid_search
best_ridge_model = grid_search.best_estimator_.named_steps['ridge']
regression_weights = best_ridge_model.coef_

print("Learned regression weights:", regression_weights.shape)
import pickle

# Save regression weights to a pickle file
with open('gemma_regression_weights.pkl', 'wb') as f:
    pickle.dump(regression_weights, f)

# Get predictions on test set
X_test_scaled = grid_search.best_estimator_.named_steps['standardscaler'].transform(X_test)
Y_pred = grid_search.best_estimator_.named_steps['ridge'].predict(X_test_scaled)

# Calculate R and R^2 values for each feature
r_values = []
r2_values = []

for i in range(Y_test.shape[1]):
    # Calculate correlation coefficient (R)
    r = np.corrcoef(Y_test[:,i], Y_pred[:,i])[0,1]
    r_values.append(r)
    
    # Calculate R^2
    r2 = r**2
    r2_values.append(r2)

# Convert to numpy arrays
r_values = np.array(r_values)
r2_values = np.array(r2_values)

print(f"Mean R value across features: {r_values.mean():.3f}")
print(f"Mean R^2 value across features: {r2_values.mean():.3f}")
print(f"\nR value range: [{r_values.min():.3f}, {r_values.max():.3f}]")
print(f"R^2 value range: [{r2_values.min():.3f}, {r2_values.max():.3f}]")

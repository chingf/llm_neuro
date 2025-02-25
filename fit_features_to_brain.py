import numpy as np
import sys
import gc
import joblib
import time
import torch
from configs import engram_dir, allstories
import os
import ridge_utils.npp
from ridge_utils.util import make_delayed
from ridge_utils.dsutils import make_word_ds
from ridge_utils.DataSequence import DataSequence
import pickle
import warnings
import argparse
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.components import TransformerBlock
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
import torch
from huggingface_hub import login
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from configs import huggingface_token

if not sys.warnoptions:
    warnings.simplefilter("ignore")
login(token=huggingface_token)
device = "cuda" if torch.cuda.is_available() else "cpu"
box_dir = os.path.join(engram_dir, 'huth_box/')

## Argument parsing
parser = argparse.ArgumentParser(description='Fit LLM activations to brain responses')
parser.add_argument('--brain_region', type=str, required=True,
                   help='Brain region to analyze (e.g. "broca")')
parser.add_argument('--model_layer', type=int, required=True,
                   help='Layer of the LLM to extract activations from')
args = parser.parse_args()

with open(os.path.join('pickles', 'voxel_indices', f'{args.brain_region}.pkl'), 'rb') as f:
    selected_features = pickle.load(f)
model_layer = args.model_layer

# Parameters for aligning LLM activations to brain responses
trim_start = 10 # Trim 50 TRs off the start of the story
trim_end = 5 # Trim 5 off the back
ndelays = 4 # We use 4 FIR delays (2 seconds, 4 seconds, 6 seconds, 8 seconds)
delays = range(1, ndelays + 1)

## Load text data
grids = joblib.load(os.path.join(box_dir, "grids_huge.jbl")) # Load TextGrids containing story annotations
trfiles = joblib.load(os.path.join(box_dir, "trfiles_huge.jbl")) # Load TRFiles containing TR information
wordseqs = make_word_ds(grids, trfiles)
for story in wordseqs.keys():
    wordseqs[story].data = [i.strip() for i in wordseqs[story].data]
print("Loaded text data")

## Load participant responses
response_path = os.path.join(box_dir, 'responses', 'full_responses', 'UTS03_responses.jbl')
resp_dict = joblib.load(response_path)
to_pop = [x for x in resp_dict.keys() if 'canplanetearthfeedtenbillionpeoplepart' in x]
for story in to_pop:
    del resp_dict[story]
train_stories = list(resp_dict.keys())
train_stories = [t for t in train_stories if t != "wheretheressmoke"]
test_stories = ["wheretheressmoke"]
print("Loaded participant responses")

# Load LLM
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
release = "gemma-scope-2b-pt-res-canonical"
sae_id = f"layer_{model_layer}/width_16k/canonical"
sae = SAE.from_pretrained(release, sae_id)[0].to(device)
with open(os.path.join("pickles", "dead_sae_unit_indices", f"gemma2b_L{model_layer}.pkl"), "rb") as f:
    alive_sae_units = pickle.load(f)['alive_units']

def get_llm_responses(stories):
    llm_responses = []
    for story in stories:
        ws = wordseqs[story]
        text_data = ws.data
        tokenized_story, word_boundaries = find_word_boundaries(text_data, tokenizer)
        hook_key = f'blocks.{model_layer}.hook_resid_post'
        with torch.no_grad():
            _, cache = model.run_with_cache(
                torch.tensor(tokenized_story).to(device),
                prepend_bos=True,
                names_filter=lambda name: name==hook_key,
            )
        llm_response = cache[hook_key][0, word_boundaries, :]
        with torch.no_grad():
            llm_response = sae.encode(llm_response)
        llm_response = llm_response[:, alive_sae_units]
        llm_data_seq = DataSequence(llm_response.cpu().numpy(), ws.split_inds, ws.data_times, ws.tr_times)
        interp_llm_response = llm_data_seq.chunksums('lanczos', window=3)
        interp_llm_response = ridge_utils.npp.zs(interp_llm_response[trim_start:-trim_end])
        llm_responses.append(interp_llm_response)
    llm_responses = np.vstack(llm_responses)
    del cache
    torch.cuda.empty_cache()
    return llm_responses

llm_train_responses = get_llm_responses(train_stories)
llm_test_responses = get_llm_responses(test_stories)[40:]
# Add FIR delays to LLM responses
delRstim = make_delayed(llm_train_responses, delays)
delPstim = make_delayed(llm_test_responses, delays)

# Package brain responses
Rresp = np.vstack([resp_dict[story] for story in train_stories])
Presp = np.vstack([resp_dict[story][40:] for story in test_stories])

# Run Regression
X_train = delRstim.astype(np.float32)
X_test = delPstim.astype(np.float32)
Y_train = Rresp[:, selected_features].astype(np.float32)
Y_test = Presp[:, selected_features].astype(np.float32)
alphas = np.logspace(5, 7, 15)  # Regularization strengths

del llm_train_responses, llm_test_responses
del delRstim, delPstim
del Rresp, Presp
del model, tokenizer, sae
del resp_dict, wordseqs
gc.collect()
torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a pipeline with standard scaling and ridge regression
ridge_pipeline = make_pipeline(
    StandardScaler(),  # Standardize features by removing the mean and scaling to unit variance
    Ridge()  # Ridge regression
)
param_grid = {'ridge__alpha': alphas}
grid_search = GridSearchCV(  # cross-validation with parallel processing
    ridge_pipeline, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=5
)

# Fit the model
print("Fitting model...")
start_time = time.time()
grid_search.fit(X_train, Y_train)
end_time = time.time()


# Get predictions on test set
X_test_scaled = grid_search.best_estimator_.named_steps['standardscaler'].transform(X_test)
Y_pred = grid_search.best_estimator_.named_steps['ridge'].predict(X_test_scaled)
r_values = []
r2_values = []
for i in range(Y_test.shape[1]):
    r = np.corrcoef(Y_test[:,i], Y_pred[:,i])[0,1]
    r_values.append(r)
    r2 = r**2
    r2_values.append(r2)
r_values = np.array(r_values)
r2_values = np.array(r2_values)
print(f"Mean R value across features: {r_values.mean():.3f}")
print(f"Mean R^2 value across features: {r2_values.mean():.3f}")
print(f"\nR value range: [{r_values.min():.3f}, {r_values.max():.3f}]")
print(f"R^2 value range: [{r2_values.min():.3f}, {r2_values.max():.3f}]")

# Save estimator to pickle
best_alpha = grid_search.best_params_['ridge__alpha']
print(f"Best alpha: {best_alpha}")
print(f"Model fitting took {(end_time - start_time) / 60:.2f} minutes")
best_ridge_model = grid_search.best_estimator_.named_steps['ridge']
model_name = 'gemma2Bfeatures'
pkl_filename = f'{args.brain_region}_{model_name}_L{model_layer}.pkl'
with open(os.path.join('pickles', 'regression_weights_sae', pkl_filename), 'wb') as f:
    results = {
        'best_estimator': grid_search.best_estimator_,
        'test_r_values': r_values,
        'best_alpha': best_alpha,
    }
    pickle.dump(results, f)


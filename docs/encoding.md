# Encoding texts and loading cached activations

ConceptExplorer analyzes existing scores. The upstream UCE package supplies
text encoding and the concept dictionary. Use the same model revision for both.

## Encode with UCE-Gemma-80k

Install the model's dependencies separately from the analysis package:

```bash
pip install -e '.[notebooks]'
```

Download the pinned model package and install its `requirements.txt`:

```python
from huggingface_hub import snapshot_download

revision = '040d2ce65439130ec211caf3349aca0e0dc53b2a'
model_dir = snapshot_download('klpeng/uce-gemma-80k', revision=revision)
print(model_dir)  # Then run: pip install -r <model_dir>/requirements.txt
```

Load the model and encode documents:

```python
import sys
import torch
from scipy import sparse
from uce_concept_explorer import ConceptExplorer

sys.path.insert(0, model_dir)
from uce_gemma import UCEGemma

device = 'cuda' if torch.cuda.is_available() else (
    'mps' if torch.backends.mps.is_available() else 'cpu'
)
model = UCEGemma.from_pretrained(model_dir, device=device)
texts = ['The election results were announced.', 'A poem about the sea.']
Z = model.encode(texts, batch_size=16)
E = ConceptExplorer(Z, texts, model.concept_dict)
sparse.save_npz('uce_scores.npz', Z)
```

The encoder retains raw top-k scores, with unreturned concepts represented as
sparse zeros. ConceptExplorer constructs binary activations from those scores.
On MPS, start with a small batch size to limit shared-memory use. Larger batches
and long documents can increase padding and memory pressure substantially.

## Load cached activations without model weights

UCE supplies standalone Python modules. Download the dictionary runtime modules,
then load metadata from the same revision as your activations:

```python
import sys
from huggingface_hub import snapshot_download
from scipy import sparse
from uce_concept_explorer import ConceptExplorer

repo_id = 'klpeng/uce-gemma-80k'
revision = '040d2ce65439130ec211caf3349aca0e0dc53b2a'
runtime_dir = snapshot_download(
    repo_id, revision=revision,
    allow_patterns=['uce_gemma.py', 'concept_metadata.py'],
)
sys.path.insert(0, runtime_dir)
from uce_gemma import load_concept_dict

concept_dict = load_concept_dict(repo_id, revision=revision)
Z = sparse.load_npz('uce_scores.npz')
# texts must be loaded in exactly the order used for encoding.
E = ConceptExplorer(Z, texts, concept_dict)
```

Dictionary keys identify matrix columns. Historical `feature_id` metadata does
not replace those keys. Optional fidelity fields may be missing or `None`.
The upstream loader supports `include_fidelity=False` and local model directories;
see the [UCE dictionary API](https://huggingface.co/klpeng/uce-gemma-80k#concept-dictionary).

Keep the texts, concept dictionary, model revision, and preprocessing settings
with saved activations. The [evaluation scripts](../evaluation/README.md) provide
splitting and batch encoding for the repository's six evaluation datasets.

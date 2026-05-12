---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Who is standing near me?
- text: Tell me What objects are on the wall.
- text: Read the text on this T-shirt.
- text: Read the text in this book.
- text: What does the screen say?
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-MiniLM-L6-v2
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 4 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                    |
|:------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | <ul><li>'Can you identify the items in my vicinity?'</li><li>'What objects am I looking at right now?'</li><li>'Scan the area and tell me What objects are here.'</li></ul> |
| 1     | <ul><li>'Who am I looking at?'</li><li>'Who is in my vicinity?'</li><li>'Tell me who is present.'</li></ul>                                                                 |
| 2     | <ul><li>'What does this document say?'</li><li>'Read the label on this item.'</li><li>'Tell me what is written here.'</li></ul>                                             |
| 3     | <ul><li>'Describe the environment.'</li><li>'What is in front of me?'</li><li>'Help me navigate this space.'</li></ul>                                                      |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Who is standing near me?")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 3   | 5.5738 | 11  |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 54                    |
| 1     | 56                    |
| 2     | 177                   |
| 3     | 18                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0013 | 1    | 0.4596        | -               |
| 0.0655 | 50   | 0.2131        | -               |
| 0.1311 | 100  | 0.0428        | -               |
| 0.1966 | 150  | 0.0139        | -               |
| 0.2621 | 200  | 0.0064        | -               |
| 0.3277 | 250  | 0.0052        | -               |
| 0.3932 | 300  | 0.0032        | -               |
| 0.4587 | 350  | 0.0019        | -               |
| 0.5242 | 400  | 0.0027        | -               |
| 0.5898 | 450  | 0.0036        | -               |
| 0.6553 | 500  | 0.0023        | -               |
| 0.7208 | 550  | 0.002         | -               |
| 0.7864 | 600  | 0.0012        | -               |
| 0.8519 | 650  | 0.0013        | -               |
| 0.9174 | 700  | 0.0012        | -               |
| 0.9830 | 750  | 0.0013        | -               |

### Framework Versions
- Python: 3.11.9
- SetFit: 1.1.3
- Sentence Transformers: 5.1.0
- Transformers: 4.56.0
- PyTorch: 2.10.0+cu130
- Datasets: 4.6.1
- Tokenizers: 0.22.0

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
## FinDVer
The data and code for the paper [FinDVer: Explainable Claim Verification over Long and Hybrid-Content Financial Documents](https://arxiv.org/abs/2411.05764). 
**FinDVer** is a comprehensive benchmark specifically designed to evaluate the explainable claim verification capabilities of LLMs in the context of understanding and analyzing long, hybrid-content financial documents.

## FinDVer Dataset
All the data examples were divided into three subsets:

- **FDV-IE** (information extraction, `ie`), which involves extracting information from both textual and tabular content within a long-context document.
- **FDV-MATH** (numerical reasoning, `numeric`), which necessitates performing calculations or statistical analysis based on data within the document.
- **FDV-KNOW** (knowledge-intensive reasoning, `knowledge`), which requires integrating external domain-specific knowledge or regulations for claim verification.

For each subset, we provide the *testmini* and *test* splits. 

The dataset is provided in the `data` folder and contains the following attributes:

```
{
    "example_id": [string] The example id
    "subset": [string] The subset of the example (FDV-IE, FDV-MATH, FDV-KNOW)
    "statement": [string] The claim to be verified
    "explanation": [list] step-by-step explanation of the claim verification, which is first annotated by human annotators and then proofread by GPT-4o
    "entailment_label": [bool] The entailment label of the claim verification
    "relevant_context": [list] The indices of the relevant context in the financial report
    "report": [string] The financial report filename, which is in the `reports` folder
}
```

## Experiments
### Environment Setup
The code is tested on the following environment:
- python 3.11.5
- CUDA 12.1, PyTorch 2.1.1
- run `pip install -r requirements.txt` to install all the required packages

### LLM Inference on DocMath-Eval
We provide inference scripts for running various LLMs on FinDVer dataset:
** 1. For RAG setting:**
- `scripts/inference/retrieval.sh` for running the retriever models to retrieve the top-n claim-relevant evidence
- `scripts/inference/main_api.sh` for running proprietary LLMs. Note that we developed a centralized API proxy to manage API calls from different organizations and unify them to be compatible with the OpenAI API. If you use the official API platform, you will need to make some modifications.
- `scripts/inference/main_vllm.sh` for running all other open-sourced LLMs that are reported in the paper and supported by the [vLLM](https://github.com/vllm-project/vllm) framework
- `scripts/inference/run_rag_analysis.sh` for running the ablation study of RAG setting

** 2. For Long-context setting:**
- `scripts/inference/longcontext_*.sh` for running the long-context setting with either proprietary or open-sourced LLMs

### Automated Evaluation
We develop a heuristic-based method to automatically evaluate the accuracy of CoT and PoT outputs:
- `scripts/evaluation/retriever_recall.sh` for evaluating the retriever recall
- `scripts/evaluate_main.sh` for evaluating Direct Output and CoT outputs
- `scripts/evaluation/evaluate_rag_analysis.sh` for evaluating the ablation study of RAG setting

### Model Output
We provide all the model outputs reported in the paper at [Google Drive](https://drive.google.com/drive/folders/1QwPb8vwjcaBNP0tHGsygXIqyqWxqtaAn?usp=sharing), specifically:
- `*_outputs/retriever_output`: The top-n claim-relevant evidence retrieved by the retriever models
- `*_outputs/rag`: The CoT and Direct Output output from all the evaluated LLMs on the RAG setting (openai embedding-3-large & top-10)
- `*_outputs/oracle`: The CoT and Direct Output output from all the evaluated LLMs on the oracle setting
- `*_outputs/long_context`: The CoT and Direct Output output from all the evaluated LLMs on testmini set on the long-context setting
- `*_outputs/rag_analysis`: The CoT and Direct Output output from all the evaluated LLMs on the ablation study of RAG setting on the testmini set 


### Test Set Evaluation
To get the results on the test set, please send your result json file to [this email](mailto:yilun.zhao@yale.edu). The result json file should at least include these features:

```
[
    {
        "example_id": [string] The question id
        "output_label": [bool] The model output, True for entailment and False for non-entailment
    }
]
```

## Contact
For any issues or questions, kindly email us at: Yilun Zhao (yilun.zhao@yale.edu).

## Citation

If you use the our benchmark in your work, please kindly cite the paper:

```
@inproceedings{zhao-etal-2024-findver,
    title = "{F}in{DV}er: Explainable Claim Verification over Long and Hybrid-content Financial Documents",
    author = "Zhao, Yilun  and
      Long, Yitao  and
      Jiang, Tintin  and
      Wang, Chengye  and
      Chen, Weiyuan  and
      Liu, Hongjun  and
      Tang, Xiangru  and
      Zhang, Yiming  and
      Zhao, Chen  and
      Cohan, Arman",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.818",
    doi = "10.18653/v1/2024.emnlp-main.818",
    pages = "14739--14752",
}
```

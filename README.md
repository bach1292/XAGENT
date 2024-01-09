# X-Agent
This repository contains code and data to reproduce the results in the paper: Answering XAI questions in a conversational agent setting 

## Data
Original data with pairs of paraphrases for XAI and their annotated scores (described in section 3) can be found [here](dataset/original-XAI-pairs-paraphrase-annotated.csv)

The filtered data, which is corresponding to "XAI question phrase bank" in the paper can be found [here](dataset/XAI-question-phrase-bank.csv) 




## Computational Environment

Install dependencies via conda:

```sh
conda env update -f environment.yml
conda activate xagent
pip install -e .
```

Start jupyter notebook:

```sh
jupyter notebook
```
## NLU evaluation 
   - Reproducibility NLU (Section 5) example command: ```python run_nlu.py -M simcse-dnn -Q all ```
     - Replace "simcse-dnn" by "svm", "rf", "simcse" to get the results of other models. 
     - Replace "all" by "xai" to get the results on xai question set
## Demo
<!-- X-Agent demo results for structured data (Section 6) [notebook](XAgent/X-Agent-structure.ipynb)
- X-Agent demo results for image data(Section 6) [notebook](XAgent/X-Agent-image.ipynb) -->
1) Open X-Agent demo in  [notebook](XAgent/X-Agent.ipynb)
2) Run all cells in the [notebook](XAgent/X-Agent.ipynb)
3) Have fun

Note:

<!-- - Both notebook files can be used for demo as they are the same agent, we only split to be easier to see the results --> 

- The interface rendering by local jupyter notebook looks better than rendering by github.

   
### Reference and Citation
Please refer to our work when using or discussing PIP-Net:

```
Van Bach Nguyen, Jörg Schlötterer, Christin Seifert (2023). “From Black Boxes to Conversations: Incorporating XAI in a Conversational Agent.". World Conference Explainable Artificial Intelligence (XAI).
```

BibTex citation:
```
@InProceedings{Nguyen2023_wcxai_xagent,
  author    = {Nguyen, Van Bach and Schl{\"o}tterer, J{\"o}rg and Seifert, Christin},
  booktitle = {Proc. World Conference Explainable Artificial Intelligence (XAI)},
  title     = {From Black Boxes to Conversations: Incorporating XAI in a Conversational Agent},
  year      = {2023},
}
```
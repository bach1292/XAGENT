# X-Agent
This repository contains code and data to reproduce the results in the paper: Talk to the Model: A Framework for Conversational XAI Agents 

## Data
Original data with pairs of paraphrases for XAI and their annotated scores (described in section 3) can be found [here](dataset/original-XAI-pairs-paraphrase-annotated.csv)

The filtered data, which is corresponding to "XAI question phrase bank" in the paper can be found [here](dataset/XAI-question-phrase-bank.csv) 




## Computational Environment

Install dependencies via conda:

```sh
conda env update -f environment.yml
conda activate xagent
```

Start jupyter notebook:

```sh
jupyter notebook
```
## NLU evaluation and X-Agent demo
   - Reproducibility NLU (Section 5) example command: ```python run_nlu.py -M simcse-dnn -Q all ```
     - Replace "simcse-dnn" by "svm", "rf", "simcse" to get the results of other models. 
     - Replace "all" by "xai" to get the results on xai question set
   - Ask the X-Agent about Machine Learning models (Section 6) [notebook](XAgent/X-Agent.ipynb)
   
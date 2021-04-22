# COMP5423 - NLP Project MCQA on MCTest, RACE, DREAM Datasets 
Group Member: 
- HON Chi Ting 20104142G 
- CHIU Ka Chun 20001511G 
- FUNG Che Hei 19013111G 
- CHAN Ka Long 20024242G 

## Introduction

We are required to develop a multiple-choice document-based question answering system to select the answer from several candidates. 

Input: a document, and a question (query) 

Output: an answer (select from options)

We are provided three multiple-choice document-based question answering dataset to
evaluate our QA system, i.e., MCTest, RACE, and DREAM. 

## Requirements
### Python packages
- Pytorch
- transformers (formerly known as pytorch-pretrained-bert and pytorch-transfomers)
- Python 3.69 

## Usage
### Get datasets
1. All three MCQA datasets needed to be put inside the folder "/content/nlp_MCQA_project/data". If you don't acquire datasets, please download it from the links below:

- MCTest: https://mattr1.github.io/mctest/data.html
- RACE: https://www.cs.cmu.edu/~glai1/data/race/
- DREAM: https://dataset.org/dream/

Folder name has to be 'MCTEST', 'RACE' and 'DREAM' (case-sensitive) respectively inside folder "/content/nlp_MCQA_project/data"

### Get pre-trained pytorch ALBERT
2. Since training an ALBERT model consumes lots of resources (GPU & time), you may download the pre-trained ALBERT model ("pytorch_model.bin") from below so that you can have a fine-tuned model for prediction:

• Pre-trained ALBERT-Large-v2 model: https://huggingface.co/albert-large-v2/tree/main

And put it inside the folder "/content/nlp_MCQA_project/tmp" and rename it as "pytorch_model_albert-large-v2.bin"

### Training
3. Or if you have sufficient GPU resource, you may train to fine-tune ALBERT model by setting parameters['do_train']=True in 'run_bert.py', then execute the following command:

```
python run_bert.py
```

One note: the effective batch size for training is important, which is the product of three variables: 
• BATCH_SIZE_PER_GPU
• NUM_OF_GPUs
• GRADIENT_ACCUMULATION_STEPS

It is recommended to be at least higher than 12 and 24 would be great.
For ALBERT-Large, 16 GB GPU (which is the maximum memory size for Cloud GPU in Google Colab) cannot hold a single batch since each data sample is composed of four choices which comprise of 4 sequences of 512 max_sequence_length. So in order to put a single batch to a 16 GB GPU, we need to decrease the max_sequence_length from 512 to some number smaller, although this will degrade the performance by a little.

### Testing
4. To test the ALBERT model, you may train BERT model (including base and large versions) by setting parameters['do_eval']=True in 'run_bert.py', then execute the following command:

```
python run_bert.py
```


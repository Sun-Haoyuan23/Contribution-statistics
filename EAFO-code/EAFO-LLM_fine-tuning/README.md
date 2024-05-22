A Method on Searching Better Activation Functions
=================================================
Large Language Model (LLM) Fine-tuning Task
=================================================
This part is completely based on https://github.com/eric-mitchell/direct-preference-optimization. We thank their works.
## Installation


Create and activate the environment :

```bash
cd ./EAFO-LLM_fine-tuning
```

```bash
conda env create -f environment.yaml
```

```bash
conda activate DPO
```

```bash
git clone https://github.com/eric-mitchell/direct-preference-optimization.git
```

## Training

In accordance with the guidelines outlined in ["DPO document"](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#dpo-direct-preference-optimization), the prescribed procedure involves executing SFT, followed by the implementation of DPO.

**GELU**

Set up the relevant configs and follow the order in ["DPO document"](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#dpo-direct-preference-optimization) step by step to run the code of DPO.

**CRReLU**

You need to modify the relevant packages in the environment. Add the CRReLU code in Appendix D.1 of the paper to ./anaconda3/envs/DPO/lib/python3.10/site-packages/transformers/activations.py and then modify the activation function components of the model by replacing them with CRReLU. Subsequent steps are the same as previous GELU steps, and you can follow the instructions in ["DPO document"](https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file#dpo-direct-preference-optimization) to execute them.


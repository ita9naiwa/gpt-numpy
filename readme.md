## GPT2 inference implemenation with numpy

Implementation of [GPT-2](https://github.com/openai/gpt-2) using numpy and scipy.

This implementaion is for educational purpose. 
I have tried to make the code as simple as possible, so that it can be easily understood.
I thought I can implement inference code without using any deep learning framework. 
I hope this code will help you understand how GPT works.

### Usage
```bash
# download model. this script is brought from ggml.
bash download-model.sh 117M

# run inference
python3 run.py 
```

or see `example.ipynb` for usage.

### The original transformers from scratch for machine-translation

##### I'm convinced that the best way to understand a machine learning model is to code it from scratch and that was the principal interest of this project.

- You can read the original paper via the link : https://arxiv.org/abs/1706.03762

- This repo contains my implementation of the original transformer model (Attention is all you need). I was inspired by this wonderful blog article of Peter Bloem : https://peterbloem.nl/blog/transformers

- Make sure you install the requirements before starting. You can create a venv and then execute this command in your terminal : pip install -r requirements.txt

- I train my model on the english-to-french dataset available on Kaggle via the link : https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench . Download it and put the data folder on your working directory. Because I don't have GPU and a lot of vRAM, I just trained my model on a half of all data. See the notebook machine-translation-with-transformers.ipynb for more details

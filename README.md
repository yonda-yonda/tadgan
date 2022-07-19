# tadgan with PyTorch
[TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks](https://arxiv.org/abs/2009.07769)

Implemented with PyTorch('1.11.0+cu113')

## dataset
sample ipynb use telemanom.
```sh
curl -O https://github.com/khundman/telemanom/labeled_anomalies.csv
curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
```
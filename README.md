# tadgan with PyTorch
[TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks](https://arxiv.org/abs/2009.07769)

Implemented with PyTorch('1.11.0+cu113')

## dataset
sample ipynb use telemanom.
```sh
curl -O https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
curl -O https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip
```

About telemanom, see https://github.com/khundman/telemanom/.
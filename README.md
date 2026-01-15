# ASAP

This repository contains the implementation of the approach proposed in the paper "*ASAP: A Self-Interpretable Graph Learning Approach for Accelerating Provenance-Based Audit Analysis*".

## Environment Setup
We recommend creating a dedicated conda environment for this project.
```bash
conda create -n asap python=3.9 -y
conda activate asap
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric==2.3.1
pip install pyg-lib==0.2.0 torch-scatter==2.1.2 torch-sparse==0.6.17 \
  torch-cluster==1.6.1 torch-spline-conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

## Dataset

This project uses data from the [DARPA Transparent Computing (TC) Program](https://github.com/darpa-i2o/Transparent-Computing/blob/master/README-E3.md).


## Running the Code
```
python src/main.py -d e3cadets -a nginx_backdoor2 --device 0
```

We provide preprocessed datasets and pre-trained models for reproducibility. You can download them from [Google Drive – ASAP_DATA](https://drive.google.com/drive/folders/1DNkLNVSbT7aoA_FrL-505pplDJO7TRFq?usp=share_link). After downloading, update the `artifact_dir` in `src/utils/config.py`:

```
artifact_dir = "/full/path/to/ASAP_DATA"
```

<table style="text-align: center">
  <caption>Summary of datasets, including training/testing splits and attack scenarios.</caption>
  <thead>
    <tr>
      <th>Datasets</th>
      <th>Platform</th>
      <th>Training Data</th>
      <th>Test Data</th>
      <th>Attack Scenario</th>
      <th>[-d]</th>
      <th>[-a]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Cadets</td>
      <td rowspan="2">FreeBSD 11.0</td>
      <td rowspan="2">2018-04-10/11</td>
      <td>2018-04-12</td>
      <td>Nginx Backdoor</td>
      <td>e3cadets</td>
      <td>nginx_backdoor2</td>
    </tr>
    <tr>
      <td>2018-04-13</td>
      <td>Nginx Backdoor</td>
      <td>e3cadets</td>
      <td>nginx_backdoor3</td>
    </tr>
    <tr>
      <td rowspan="2">Theia</td>
      <td rowspan="2">Ubuntu 12.04</td>
      <td rowspan="2">2018-04-09/11/13</td>
      <td>2018-04-10</td>
      <td>Firefox Backdoor</td>
      <td>e3theia</td>
      <td>firefox_backdoor</td>
    </tr>
    <tr>
      <td>2018-04-12</td>
      <td>Browser Extension</td>
      <td>e3theia</td>
      <td>browser_extension</td>
    </tr>
    <tr>
      <td rowspan="3">Trace</td>
      <td rowspan="3">Ubuntu 14.04</td>
      <td rowspan="3">2018-04-09/11</td>
      <td>2018-04-10</td>
      <td>Firefox Backdoor</td>
      <td>e3trace</td>
      <td>firefox_backdoor</td>
    </tr>
    <tr>
      <td>2018-04-12</td>
      <td>Browser Extension</td>
      <td>e3trace</td>
      <td>browser_extension</td>
    </tr>
    <tr>
      <td>2018-04-13</td>
      <td>Pine Backdoor</td>
      <td>e3trace</td>
      <td>pine_backdoor</td>
    </tr>
    <tr>
      <td>Clearscope</td>
      <td>Android 6.0.1</td>
      <td>2018-04-10/12/13</td>
      <td>2018-04-11</td>
      <td>Firefox Backdoor</td>
      <td>e3clearscope</td>
      <td>firefox_backdoor</td>
    </tr>
  </tbody>
</table>
# Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling

This repository contains the PyTorch implementation for the paper [Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling](https://arxiv.org/abs/2210.12156).
This work has been accepted at the [International Conference on Machine Learning](https://icml.cc/), 2023. 

## Set up environment

### Environment
Run the following commands to create a conda environment:
```bash
conda create -n MulEHR python=3.8
source activate MulEHR
pip install -r requirements.txt
```

### Data 
We uilize open-source EHR [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) to conduct experiment. This dataset is a restricted-access resource. To access the files, you must be a credentialed user and sign the data use agreement (DUA) for the project. Because of the DUA, we cannot provide the data directly.

After download MIMIC-III, we obtain time series and clinical notes following: 

1. Generate time serise data following [MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks). Note, there are five tasks in the [MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks): in-hospital-mortality, decompensation, length-of-stay, phenotyping and multitask. We conduct experiments on in-hospital-mortality and phenotyping, which are more important based on clinicans' suggestion. Effetiveness of our model on other tasks are leveage as furture works.
2. Extract clinical note data following [ClinicalNotesICU](https://github.com/kaggarwal/ClinicalNotesICU)
3. Process regular time series, irregular time series and clinical notes.
For example, for 48 IHM task, runs

```bash
python preprocess.py
```
with defualt setting. 

#### Build your own data
To build your own task, you need a dataset of a list of instances. For each instance, the following information is required:
- `name`: ID number.
- `irg_ts`: Irregular time series matrix, which is a d_m x l_ts np.array. Here, d_m is the number of features, and l_ts is the total number of distinct time points.
- `irg_ts_mask`: Irregular time series mask matrix, the same shape as irg_ts. When there is an existing value in the corresponding position of irg_ts, the mask value is 1; otherwise, it is 0.
- `reg_ts`: Imputed regular time series matrix, which is a (d_m x 2) x l_reg np.array. Here, (d_m x 2) is the number of features with corresponding masks, and l_reg is the length of the predefined time interval.
- `ts_tt`: A list of time points of irregular time series, whose length is l_ts.
- `text_data`: A list of clinical notes, whose length is l_txt, the number of clinical notes a patient has.
- `text_time_to_end`: A list of time points of irregular clinical notes, whose length is l_txt.
- `label`: The predicted output.


### Experiment
Run following bash file to conduct experiment:
```bash
sh run.sh
```



## Citation

If you found this repository useful, please consider cite our paper:

```bibtex
@misc{zhang2022improving,
      title={Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling}, 
      author={Xinlu Zhang and Shiyang Li and Zhiyu Chen and Xifeng Yan and Linda Petzold},
      year={2022},
      eprint={2210.12156},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements

We would like to acknowledge the following open-source projects that were used in our work:

- [MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) for generating the time series data from the MIMIC-III dataset.

- [ClinicalNotesICU](https://github.com/kaggarwal/ClinicalNotesICU) for extracting the clinical notes data from the MIMIC-III dataset.

The use of these open-source projects has been instrumental in our research and we are grateful for the contributions made by their authors.




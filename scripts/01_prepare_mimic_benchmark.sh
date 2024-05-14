echo 'Extracting Subjects'
python -m mimic3-benchmarks.mimic3benchmark.scripts.extract_subjects ../physionet.org/files/mimiciii/1.4/ data_mimic/root/

echo 'Validating Events'
python -m mimic3-benchmarks.mimic3benchmark.scripts.validate_events data_mimic/root/

echo 'Extracting Episodes from subjects'
python -m mimic3-benchmarks.mimic3benchmark.scripts.extract_episodes_from_subjects data_mimic/root/

echo 'Train Test Split'
python -m mimic3-benchmarks.mimic3benchmark.scripts.split_train_and_test data_mimic/root/


echo 'Creating different task'
python -m preprocess.preprocess_mimic_iii data_mimic/root/ data_mimic/in-hospital-mortality-24/

echo 'Train Val Split'
python -m mimic3-benchmarks.mimic3models.split_train_val data_mimic/in-hospital-mortality-24/
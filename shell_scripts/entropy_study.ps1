conda activate nlp
cd $ST_4

python .\inference_scripts\entropy_study_mean_head_agreg.py --batch_size 4
python .\inference_scripts\entropy_study_sum_head_agreg.py --batch_size 4
python .\inference_scripts\entropy_study_mean_evw_agreg.py --batch_size 4
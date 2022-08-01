conda activate nlp
cd $ST_4


python .\inference_scripts\cls_study_sum_head_agreg.py --batch_size 4
python .\inference_scripts\cls_study_mean_head_agreg.py --batch_size 4
python .\inference_scripts\cls_study_mean_evw_agreg.py --batch_size 4
python .\inference_scripts\cls_study_cls_sep_through_layers.py --batch_size 4
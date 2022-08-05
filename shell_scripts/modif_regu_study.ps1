conda activate nlp
cd $ST_4

python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.0
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.001 --modif

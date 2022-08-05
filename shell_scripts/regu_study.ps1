conda activate nlp
cd $ST_4

python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.0
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.001
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.0015
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.002
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.0025
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.003
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.0035



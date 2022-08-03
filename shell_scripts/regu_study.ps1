conda activate nlp
cd $ST_4

python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.0
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.001
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.002
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.003
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.004
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.005
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.006
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.007
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.008
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.009
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.01
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.02
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.03
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.04
python .\inference_scripts\entropy_study_regu_study.py --batch_size 4 --reg_mul 0.05




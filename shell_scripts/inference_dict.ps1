conda activate nlp

cd $ST_4

#python .\inference_script\get_pickle_inference_dict.py --reg_mul 0.005
#python .\inference_script\get_pickle_inference_dict.py --reg_mul 0.08
#python .\inference_script\get_pickle_inference_dict.py --reg_mul 0.4
#python .\inference_script\get_pickle_inference_dict.py --reg_mul 0.1
#python .\inference_script\get_pickle_inference_dict.py --reg_mul 0.004
#python .\inference_script\get_pickle_inference_dict.py --reg_mul 0.003
#python .\inference_script\get_pickle_inference_dict.py --reg_mul 0.002
python .\inference_scripts\get_pickle_inference_dict.py --reg_mul 0.001
#python .\inference_script\get_pickle_inference_dict.py

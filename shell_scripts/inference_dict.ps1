conda activate nlp

cd $ST_4

python .\get_pickle_inference_dict.py --reg_mul 0.4
python .\get_pickle_inference_dict.py --reg_mul 0.1
python .\get_pickle_inference_dict.py --reg_mul 0.004
python .\get_pickle_inference_dict.py --reg_mul 0.003
python .\get_pickle_inference_dict.py --reg_mul 0.002
python .\get_pickle_inference_dict.py --reg_mul 0.001
python .\get_pickle_inference_dict.py

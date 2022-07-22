#!/usr/bin/env bash
#OAR -l {host='igrida-abacus3.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus13.irisa.fr' OR host='igrida-abacus18.irisa.fr' OR host='igrida-abacus21.irisa.fr'}/nodes=1/gpu_device=1,walltime=24:00:00
#OAR -O /srv/tempdd/dunguyen/RUNS/loic/%jobid%.out.log
#OAR -E /srv/tempdd/dunguyen/RUNS/loic/%jobid%.err.log

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh

module load spack/cuda/11.3.1
module load spack/cudnn/8.0.4.30-11.0-linux-x64

conda deactivate
source $VENV/bert/bin/activate

EXEC_FILE=regularize_training_bert.py

echo
echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at `date +"%T, %d-%m-%Y"`

# if --reg_lay == -1 ==> regularize all the model
# /!\ if --reg-lay == i we regularize the layer i+1

python $EXEC_FILE -e 3 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.1  --version debug --exp --reg_lay 2


echo Script ended
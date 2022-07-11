#!/usr/bin/env bash
#OAR -l {host='igrida-abacus3.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus13.irisa.fr' OR host='igrida-abacus18.irisa.fr' OR host='igrida-abacus21.irisa.fr'}/nodes=1/gpu_device=1,walltime=24:00:00
#OAR -O /srv/tempdd/dunguyen/RUNS/%jobid%.out.log
#OAR -E /srv/tempdd/dunguyen/RUNS/%jobid%.err.log

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

python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.0  --version mul=0.0_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.2  --version mul=0.2_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.3  --version mul=0.3_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.4  --version mul=0.4_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.5  --version mul=0.5_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.6  --version mul=0.6_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.7  --version mul=0.7_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.8  --version mul=0.8_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 0.9  --version mul=0.9_layer=3 --exp --reg_lay 3
python $EXEC_FILE -e 50 -b 32 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_reg_nli --reg_mul 1.0  --version mul=1.0_layer=3 --exp --reg_lay 3

echo Script ended
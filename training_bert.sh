#!/usr/bin/env bash
#OAR -l {host='igrida-abacus3.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus13.irisa.fr' OR host='igrida-abacus18.irisa.fr' OR host='igrida-abacus21.irisa.fr'}/nodes=1/gpu_device=1,walltime=24:00:00
#OAR -O /srv/tempdd/dunguyen/RUNS/%jobid%.out.log
#OAR -E /srv/tempdd/dunguyen/RUNS/%jobid%.err.log

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh

module load spack/cuda/11.3.1
module load spack/cudnn/8.0.4.30-11.0-linux-x64

source $VENV/nlp/bin/activate

EXEC_FILE=src/training_bert.py

echo
echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at `date +"%T, %d-%m-%Y"`
python $EXEC_FILE -e 50 -b 512 -d $RUNDIR/loic/dataset -s $RUNDIR/loic/logs --experiment bert_nli --version run=1

echo Script ended
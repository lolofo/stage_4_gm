#!/usr/bin/env bash
#OAR -l {host='igrida-abacus3.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus4.irisa.fr' OR host='igrida-abacus13.irisa.fr' OR host='igrida-abacus18.irisa.fr' OR host='igrida-abacus21.irisa.fr'}/nodes=1/gpu_device=1,walltime=24:00:00
#OAR -O /srv/tempdd/dunguyen/RUNS/%jobid%.out.log
#OAR -E /srv/tempdd/dunguyen/RUNS/%jobid%.err.log

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh

module load spack/cuda/11.3.1
module load spack/cudnn/8.0.4.30-11.0-linux-x64

source $VENV/nlp/bin/activate

EXEC_FILE=src/regularization_delta_entropy.py

echo
echo =============== RUN ${OAR_JOB_ID} ===============
echo Run $EXEC_FILE at `date +"%T, %d-%m-%Y"`
python $EXEC_FILE -o $RUNDIR -b 512 -e 50 --lambda 0.0 --vectors glove.840B.300d -m exp --name $OAR_JOB_NAME --version lambda1=0

echo Script ended
#PBS -N MCC_AUC
#PBS -q default
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=16,mem=31000m,walltime=168:00:00 
#PBS -M w186f427@ku.edu,s584y458@ku.edu   
#PBS -d /projects/huanlab/wei/wrapper-16
#PBS -e /projects/huanlab/wei/results16/mcc_auc-stderr.txt
#PBS -o /projects/huanlab/wei/results16/mcc_auc-stdout.txt

# Save job specific information for troubleshooting

echo "Job ID is ${PBS_JOBID}"
echo "Running on host $(hostname)"
echo "Working directory is ${PBS_O_WORKDIR}"
echo "The following processors are allocated to this job:"
echo $(cat $PBS_NODEFILE)
echo "Job ID is ${PBS_JOBID}"
echo "Job name is MCC_AUC"
echo "Running on host $(hostname)"
echo "Working directory is ${PBS_O_WORKDIR}"
echo "The following processors are allocated to this job:"
echo $(cat $PBS_NODEFILE)
echo $PBS_ARRAYID
export CLASSPATH=/projects/huanlab/wei/weka.jar:$CLASSPATH
export CLASSPATH=/projects/huanlab/wei/RBFNetwork.jar:$CLASSPATH
export CLASSPATH=/projects/huanlab/wei/multisearch2015.10.15.jar:$CLASSPATH
module load jdk/1.8.0_31

# Run the program
echo "Start: $(date +%F_%T)"
javac -cp \* MCC_AUC.java
java -cp .:\* MCC_AUC
echo "Stop: $(date +%F_%T)"
~

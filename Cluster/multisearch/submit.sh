#PBS -N wekatest
#PBS -q default
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8,mem=15g,walltime=1024:00:00 
#PBS -M w186f427@ku.edu,s584y458@ku.edu   
#PBS -d /projects/huanlab/wei   
#PBS -e /projects/huanlab/wei/results/stderr.txt
#PBS -o /projects/huanlab/wei/results/stdout.txt
#PBS -t 8-17

# Save job specific information for troubleshooting

filename=("aLogCfsBf" "bLogCfsGreedy" "cLogCorrRank" "dLogGainRank" "eLogInfoRank" "fLog1rRank" "gLogPrinRank" "hLogReliefRank" "iLogSymmRank" "jRbfCfsBf" "kRbfCfsGreedy" "lRbfCorrRank" "mRbfGainRank" "nRbfInfoRank" "oRbf1rRank" "pRbfPrinRank" "qRbfReliefRank" "rRbfSymmRank")

echo "Job ID is ${PBS_JOBID}"
echo "ArrayID is ${PBS_ARRAYID}"
echo "Running on host $(hostname)"
echo "Working directory is ${PBS_O_WORKDIR}"
echo "The following processors are allocated to this job:"
echo $(cat $PBS_NODEFILE)
echo "Job ID is ${PBS_JOBID}"
echo "Job name is ${filename[${PBS_ARRAYID}]}"
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
javac -cp \* ${filename[${PBS_ARRAYID}]}.java 
java -cp .:\* ${filename[${PBS_ARRAYID}]}
echo "Stop: $(date +%F_%T)"
~                             

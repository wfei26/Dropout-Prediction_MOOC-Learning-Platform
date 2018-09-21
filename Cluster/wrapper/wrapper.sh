#PBS -N wekatest-wrapper
#PBS -q default
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=8,mem=1590m,walltime=160:00:00 
#PBS -M w186f427@ku.edu,s584y458@ku.edu   
#PBS -d /projects/huanlab/wei   
#PBS -e /projects/huanlab/wei/results/wrapper-stderr.txt
#PBS -o /projects/huanlab/wei/results/wrapper-stdout.txt
#PBS -t 0-5

# Save job specific information for troubleshooting

filename=("LogWrapBF" "LogWrapGS" "RFWrapBF" "RFWrapGS" "RBFWrapBF" "RBFWrapGS")

echo "Job ID is ${PBS_JOBID}"
echo "ArrayID is ${PBS_ARRAYID}"
echo "Running on host $(hostname)"
echo "Working directory is ${PBS_O_WORKDIR}"
echo "The following processors are allocated to this job:"
echo $(cat $PBS_NODEFILE)
echo "Job ID is ${PBS_JOBID}"
echo "Job name is 5"
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

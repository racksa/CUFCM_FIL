#!/bin/bash

# Submit the PBS job script to the queue, catching the job ID as we do so.
# if [ $# -eq 1 ] && [ $1 = 'express' ]
# then
# 	JOB_ID=$(qsub -q express -P exp-XXXXX run_cilia_sim.pbs)
# 	echo "Using the express queue..."
# else
# 	JOB_ID=$(qsub run_cilia_sim.pbs)
# 	echo "Using the normal queue..."
# fi
# JOB_ID=${JOB_ID:0:-4} # Throw away the .pbs extension

# Make the job-specific directory to store everything related to this job.
# LOC=$JOB_ID
# mkdir -p jobs/$LOC

# Copy the source files, along with any input files, to this directory.
# The executable will be compiled as part of the job, ensuring we have a preserved copy of the .par etc. used for a given simulation.
# Note: large input files should probably be copied as part of the job in case they take longer to copy than the job spends in the queue. 
# cp -r src $LOC/
# cp makefile $LOC/
# cp config.hpp $LOC/
# cp *.input $LOC/

# And we should be done!

# q=9
# command="python3 pyfile/bisection/bisection.py ${q} 9 0 "

q=21
nq=40
command="python3 pyfile/driver/driver.py run ${q} ${nq} 1"


# cp --attributes-only "run_cilia_sim.pbs" "pbs/run_cilia_sim${q}.pbs"
sed -e "\$a\\$command" "run_cilia_sim.pbs" > "pbs/run_cilia_sim${q}.pbs"
chmod +x "pbs/run_cilia_sim${q}.pbs"

JOB_ID=$(qsub pbs/run_cilia_sim${q}.pbs)
JOB_ID=${JOB_ID:0:-4}
echo "Using the normal queue..."
echo "Job submitted!"
echo $JOB_ID
tail -n 1 pbs/run_cilia_sim${q}.pbs
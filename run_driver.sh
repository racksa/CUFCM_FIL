#!/bin/bash
#
# Run python3 pyfile/driver/driver.py across multiple threads, with optional
# GPU split across two devices.
#
# Usage:
#   bash run_driver.sh [TOTAL_THREADS] [CUDA_DEVICE] [START_THREAD] [END_THREAD]
#
# Arguments:
#   TOTAL_THREADS  Total number of threads the job is split into  (default: 4)
#   CUDA_DEVICE    CUDA device to use for this batch               (default: 0)
#   START_THREAD   First thread index to launch (inclusive)        (default: 0)
#   END_THREAD     Last  thread index to launch (exclusive)        (default: TOTAL_THREADS)
#
# Examples:
#   bash run_driver.sh 10 0          # threads 0-9 on GPU 0
#   bash run_driver.sh 10 0 0 5 &   # threads 0-4 on GPU 0  \  split across
#   bash run_driver.sh 10 1 5 10 &  # threads 5-9 on GPU 1  /  two GPUs

TOTAL_THREADS=${1:-4}
CUDA_DEVICE=${2:-0}
START_THREAD=${3:-0}
END_THREAD=${4:-$TOTAL_THREADS}

cd "$(dirname "$0")"

for (( i=START_THREAD; i<END_THREAD; i++ )); do
    echo "Starting thread $i / $TOTAL_THREADS on CUDA device $CUDA_DEVICE"
    python3 pyfile/driver/driver.py run "$i" "$TOTAL_THREADS" "$CUDA_DEVICE" &
    sleep 5
done

wait
echo "Threads $START_THREAD–$((END_THREAD-1)) finished."

#!/bin/bash

max_jobs=4
for task in task{1..5}; do
  $task &
  while (( $(jobs -r | wc -l) >= max_jobs )); do wait -n; done
done
wait

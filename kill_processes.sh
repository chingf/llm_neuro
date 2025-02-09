#!/bin/bash

pids=$(ps aux | grep /n/home04/cfang/.conda/envs/sae/bin/python | grep -v grep | awk '{print $2}')

for pid in $pids; do
    echo "Killing process $pid"
    kill -9 $pid
done

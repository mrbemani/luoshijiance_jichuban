#!/bin/bash

while true; do
    # Delete files older than 10 minutes
    find /opt/vcr/ -type f -mmin +10 -delete
    sleep 10
done
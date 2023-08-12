#!/bin/bash

while true; do
    # Delete files older than 10 minutes
    find /ssd_disk/vcr/ -type f -mmin +10 -delete
    sleep 10
done
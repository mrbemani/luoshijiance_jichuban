#!/bin/bash

while true; do
    # Delete files older than 10 minutes
    find $VCR_PATH -type f -mmin +10 -delete
    sleep 10
done
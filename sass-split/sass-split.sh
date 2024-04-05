#!/bin/bash

SEARCH_ROOT="/home/XXX/apps/OursTracesCollection"

CMD="./process_sass_dir --dir"

find "$SEARCH_ROOT" -type d -name "sass_traces" | while read dir; do
    echo "Processing directory: $dir"
    $CMD "$dir"
done
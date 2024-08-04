#!/bin/bash
# Check if the output file argument was provided
if [ $# -eq 0 ]
then
    echo "Please provide the output file as an argument"
    return
fi

# Check if the directory of Python programs argument was provided
if [ $# -eq 1 ]
then
    echo "Please provide the directory of Python programs as an argument"
    return
fi

# Set the output file
output_file=$1
# Set the directory of Python programs
python_programs_dir=$2
# Loop through all files in the directory of Python programs
for file in $python_programs_dir/*.py
do
    # Execute the Python program and append the output to the output file
   sudo env PATH="$PATH" python $file $output_file
done

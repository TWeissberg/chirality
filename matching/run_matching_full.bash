#!/bin/bash
#
#SBATCH --job-name=matching
#SBATCH --ntasks=1
#SBATCH --time=47:59:59
#SBATCH --partition=mlgpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-53

# Define the list of commands
commands=(
  "matching_scape.py"
  "matching_tosca.py"
  "macthing_faust.py"
  "matching_smal.py"
  "matching_becos.py"
  "matching_becos_a_becos_a.py"
  "matching_becos_a_becos_h.py"
  "matching_becos_h_becos_h.py"
  "matching_becos_h_becos_a.py"
)

# Define the list of parameters
params=("diff3f" "sd+dino" "siggraph" "iccv" "gt_chirality_cat" "gt_chirality_mul")
#params=("iccv" "gt_chirality_cat" "gt_chirality_mul")

# Total combinations: 9 commands × 6 parameters = 54
index=${SLURM_ARRAY_TASK_ID}
cmd_index=$((index / 6))
param_index=$((index % 6))

cmd=${commands[$cmd_index]}
param=${params[$param_index]}

echo "Running: $cmd with parameter: $param"
python3 "./$cmd" "$param"
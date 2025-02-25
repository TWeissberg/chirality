#!/bin/bash
#
#SBATCH --job-name=matching
#SBATCH --ntasks=1
#SBATCH --time=47:59:59
#SBATCH --partition=mlgpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-26

# Define the list of commands
commands=(
  "self_matching_scape.py"
  "self_matching_tosca.py"
  "self_macthing_faust.py"
  "self_matching_smal.py"
  "self_matching_becos.py"
  "self_matching_becos_a_becos_a.py"
  "self_matching_becos_a_becos_h.py"
  "self_matching_becos_h_becos_h.py"
  "self_matching_becos_h_becos_a.py"

)

# Define the list of parameters
params=("diff3f" "sd+dino" "siggraph")

# Total combinations: 9 commands × 3 parameters = 27
index=${SLURM_ARRAY_TASK_ID}
cmd_index=$((index / 3))
param_index=$((index % 3))

cmd=${commands[$cmd_index]}
param=${params[$param_index]}

echo "Running: $cmd with parameter: $param"
python3 "./$cmd" "$param"
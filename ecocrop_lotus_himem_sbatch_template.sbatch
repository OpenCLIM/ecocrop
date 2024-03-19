#!/bin/bash
#SBATCH -p high-mem
#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH --time=48:00:00
#SBATCH --mem=800GB

# activate anaconda environment
export PATH=/home/users/xxxxxx/anaconda3/bin:$PATH # replace with path to anaconda3 bin dir on your sys
conda /home/users/xxxxxx/anaconda3/bin/activate testecocrop

# use full path to specific environment's python, in case environment
# activatation doesn't work
# replace cropind, rcp, ensmem, pf, method as desired, see README and script
/home/users/xxxxxx/anaconda3/envs/testecocrop/bin/python /path/to/ecocrop_lotus_himem.py cropind rcp ensmem pf method

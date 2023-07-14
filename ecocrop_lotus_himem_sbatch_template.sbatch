#!/bin/bash
#SBATCH -p high-mem
#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH --time=48:00:00
#SBATCH --mem=800GB

export PATH=/home/users/mattjbr/anaconda3/bin:$PATH
source /home/users/mattjbr/anaconda3/bin/activate ecocrop
/home/users/mattjbr/anaconda3/envs/ecocrop/bin/python /gws/nopw/j04/ceh_generic/matbro/ecocrop/ecocrop_lotus_himem.py cropind rcp ensmem pf method

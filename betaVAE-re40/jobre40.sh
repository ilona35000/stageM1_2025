#!/bin/bash
#SBATCH   --job=scratchre40  ##########nom arbitraire
#SBATCH --cluster=nautilus
#SBATCH --partition=gpu    #gpu ou visu
#SBATCH --time=1-00:00:00 
#SBATCH --gres=gpu:2      #[1,4]gpu ou [1,2]visu
#SBATCH --qos=short
#SBATCH  --output=output_nautilus/resultat.log
#SBATCH --error=output_nautilus/error_compile.err



source ~/.bashrc


conda activate ilona





python main.py -re 40 -m run -nn easy
python testStochastique.py
python lectre40_hdf5.py

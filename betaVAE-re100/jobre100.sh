#!/bin/bash
#SBATCH   --job=scratchre100  ##########nom arbitraire
#SBATCH --cluster=nautilus
#SBATCH --partition=gpu    #gpu ou visu
#SBATCH --time=1-00:00:00 
#SBATCH --gres=gpu:2      #[1,4]gpu ou [1,2]visu
#SBATCH --qos=short
#SBATCH  --output=output_nautilus/resultat.log
#SBATCH --error=output_nautilus/error_compile.err
#SBATCH --mem=264G


source ~/.bashrc


conda activate ilona





#python main.py -re 100 -m run -nn easy
python testStochastique.py
#python lectre100_hdf5.py

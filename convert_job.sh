#!/bin/bash
#SBATCH -J abc2_web
#SBATCH -p cyclopes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o /scratch/ralbe/ipole_abc_web/convert_%j.out
cd /scratch/ralbe/ipole_abc_web
.venv/bin/python convert.py --out web_data --workers 12

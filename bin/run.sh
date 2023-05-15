#!/bin/bash
dataset=house
for i in 0
do
	python save_model.py /Users/timgritsaev/Desktop/tabular/tabular-hpo/exp/mlp/$dataset/$i-tuning/checkpoint.pt optuna_${dataset}_$i.pt	
done

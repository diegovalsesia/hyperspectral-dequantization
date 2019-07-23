#!/bin/bash

# "prequantization" or "ccsds"
models=("prequantization")
# list of quantization step sizes to train for
Qs=("101")

this_folder=`pwd`

for model in ${models[@]}; do
	for Q in ${Qs[@]}; do
	
		# check if it is already trained or we want to retrain
		if [[ -f "$this_folder/log_dir/$model/Q$Q/start_iter" ]]; then
			start_iter=`cat "$this_folder/log_dir/$model/Q$Q/start_iter"`
		else
			start_iter=1
		fi
		
		log_dir="$this_folder/log_dir/$model/Q$Q/"
		save_dir="$this_folder/Results/$model/Q$Q/saved_models/"
		data_dir="$this_folder/Data/$model/train/Q$Q/"
		quant_filename="all_Q$Q.mat"
		test_dir="$this_folder/Data/$model/test/Q$Q/"
		test_quant_filename="sc0_Q""$Q"".mat"

		CUDA_VISIBLE_DEVICES=0 python3 "Code/$model/main.py" --start_iter $start_iter --log_dir $log_dir --save_dir $save_dir --data_dir $data_dir --quant_filename $quant_filename --Q $Q --test_dir $test_dir --test_quant_filename $test_quant_filename

	done
done

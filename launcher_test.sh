#!/bin/bash

# "prequantization" or "ccsds"
models=("ccsds")
# list of quantization step sizes to test
Qs=("61")
# list of scene filenames for test images
scenes=("sc0")
# shift along the band axis: 8 bands are used to produce an estimate and then the input is shifted by sz bands
sz=1

this_folder=`pwd`

for model in ${models[@]}; do
	for Q in ${Qs[@]}; do
		for scene in ${scenes[@]}; do
			
			echo "Scene: $scene Q: $Q"
			
			savedmodel_file="$this_folder/Results/$model/Q$Q/saved_models/G_last.pth"
			test_dir="$this_folder/Data/$model/test/Q$Q/"
			test_quant_filename="$scene""_Q""$Q"".mat"
			output_filename="$scene""_dequantized_conv_""$sz.mat"
			reconstruct_dir="$this_folder/Results/$model/Q$Q/reconstructed_test_image/"
			
			CUDA_VISIBLE_DEVICES=0 python3 "Code/$model/test_conv.py" --G_load $savedmodel_file --Q $Q --test_dir $test_dir --test_quant_filename $test_quant_filename --reconstruct_dir $reconstruct_dir --output_filename $output_filename --sz $sz

		done
	done
done

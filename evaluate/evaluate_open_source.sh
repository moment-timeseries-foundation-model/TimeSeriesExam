export CUDA_VISIBLE_DEVICES=0

python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name 'microsoft/Phi-3.5-mini-instruct'\
                    --seed 42 \
                    --max_tokens 1024 \
                    --temperature 0.0\
                    --metric 'accuracy'\
                    --output_file_path output/round_3_folder/evaluate_result.json\
                    --ts_tokenizer_name 'plain_text'\
                    --device 'cuda:0'\
                    --add_question_hint\
                    --add_concepts\
                    --add_examples\

image_cache_dir='' # your_cache_dir for image (image will be plotted and saved during inference)

python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name "microsoft/Phi-3.5-vision-instruct"\
                    --seed 42 \
                    --max_tokens 1024 \
                    --temperature 0.0\
                    --metric 'accuracy'\
                    --output_file_path output/round_3_folder/evaluate_result.json\
                    --image_cache_dir ${image_cache_dir}\
                    --ts_tokenizer_name 'image'\
                    --add_question_hint\
                    --add_concepts\
                    --add_examples\
                    --device 'cuda:0'\

python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name 'openbmb/MiniCPM-V-2_6'\
                    --seed 42 \
                    --max_tokens 1024 \
                    --temperature 0.0\
                    --metric 'accuracy'\
                    --output_file_path output/round_3_folder/evaluate_result.json\
                    --image_cache_dir ${image_cache_dir}\
                    --ts_tokenizer_name 'image'\
                    --device 'cuda:0'\
                    --add_question_hint\
                    --add_concepts\
                    --add_examples\

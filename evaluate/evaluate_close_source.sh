image_cache_dir= '' # your_cache_dir for image (image will be plotted and saved during inference)

###########GPT-4o############
python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name gpt-4o\
                    --seed 42 \
                    --max_tokens 1024 \
                    --temperature 0.0\
                    --metric 'accuracy'\
                    --output_file_path output/round_3_folder/evaluate_result.json\
                    --ts_tokenizer_name 'plain_text'\
                    --add_question_hint\
                    --add_concepts\
                    --add_examples\

python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name gpt-4o\
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

###########Claude############
python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name claude-3-5-sonnet-20240620\
                    --seed 42 \
                    --max_tokens 1024 \
                    --temperature 0.0\
                    --metric 'accuracy'\
                    --output_file_path output/round_3_folder/evaluate_result.json\
                    --ts_tokenizer_name 'plain_text'\
                    --add_question_hint\
                    --add_concepts\
                    --add_examples\

python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name "claude-3-5-sonnet-20240620"\
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

###########Gemini############
python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name gemini-1.5-pro\
                    --seed 42 \
                    --max_tokens 1024 \
                    --temperature 0.0\
                    --metric 'accuracy'\
                    --output_file_path output/round_3_folder/evaluate_result.json\
                    --ts_tokenizer_name 'plain_text'\
                    --add_question_hint\
                    --add_concepts\
                    --add_examples\

python3 evaluate/evaluate.py --data_file_path output/round_3_folder/qa_dataset.json\
                    --model_name gemini-1.5-pro\
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
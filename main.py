from question_template import (
    TEMPLATE,
    CONCEPTS
)

from utils.utils import (
    get_ts_obj_from_option,
    get_ts_obj_from_two_options,
    get_pair_ts_obj_from_option,
    calculate_noise_level,
    execute_option,
    seed_everything, 
    Option,
    TwoTSOption,
    PairTSOption,
)

from utils.error_classes import (
    AnomalyOnePeakError
)

from timeseries_curation.composer import Composer

import random 
import argparse
import json 
import os

def get_qa_pairs(template:dict, num_questions_per_option:int, ts_length:int):
    '''
    template: dict, template containing question, options, question_type, tid, difficulty, format_hint, relevant_concepts, question_hint
    num_questions_per_option: int, number of questions to generate per option. This only applies when num_draws is None for the option
    ts_length: int, length of the time-series
    '''

    qa_pairs = [] 

    for i, option in enumerate(template['options']):

        num_draws = option.num_draws
        if num_draws is None:
            num_draws = num_questions_per_option

        for _ in range(num_draws):
            if isinstance(option, Option):
                ts_obj = get_ts_obj_from_option(option)

                #get meta info
                question = template['question']
                answer = option.option_name
                question_type = template['question_type']
                tid = template['tid']
                difficulty = template['difficulty']

                #generate time-series
                ts = ts_obj.generate(ts_length)

                #we apply noise only if the snr is greater than 0
                if option.noise_snr > 0.0:
                    noise_level = calculate_noise_level(option.noise_snr, ts_obj, ts_length)
                    noise_obj = template['noise'](noise_level=noise_level)
                    ts = ts + noise_obj.generate(ts_length)

                #if ts_object has attribute transformation 
                if hasattr(ts_obj, 'transformations'):
                    try:
                        for transformation_obj in ts_obj.transformations:
                            ts = transformation_obj.transform(ts)
                    except AnomalyOnePeakError:
                        print(f'ERROR: tid {tid}. Found one peak error. This is because the time series has only one peak. Increasing the length of the time series will usually solve this issue. Skipping this sample')
                        continue
                    except Exception as e:
                        raise Exception(f'Error in template: {template["tid"]}. Error: {e}')
                        
                #shuffle the options, replace answers with actual value if the answer is an actual value of a parameter
                option_names = [execute_option(option.option_name) for option in template['options']] #[('option_name', 'actual_value')]
                answer = option_names[i] #because we are iterating over the options
                option_names = [elem[1] for elem in option_names] #option_names = ['actual_value']
                if answer[0] == answer[1]:
                    #this happens when the answer is not about a parameter value, but a fixed quantity
                    answer = answer[0]
                else:
                    #if the object is a composer, answer object id refers to the id of the object in the object list
                    if isinstance(ts_obj, Composer) and template.get('answer_object_id', None) is not None:
                        answer = round(ts_obj.ts_objects[template['answer_object_id']].__dict__[answer[0]], 2)
                    else:
                        answer = round(ts_obj.__dict__[answer[0]], 2)
                    option_names[i] = answer

                random.shuffle(option_names)

                qa_pairs.append({
                    'question': question,
                    'options': option_names,
                    'answer': answer,
                    'question_type': question_type,
                    'ts': ts,
                    'tid': tid,
                    'difficulty': difficulty,
                    'format_hint': template['format_hint'],
                    'relevant_concepts': template['relevant_concepts'],
                    'question_hint': template['question_hint']
                })

            elif isinstance(option, TwoTSOption):
                ts_obj1, ts_obj2 = get_ts_obj_from_two_options(option)

                #get meta info
                question = template['question']
                answer = option.option_name
                question_type = template['question_type']
                tid = template['tid']
                difficulty = template['difficulty']

                #generate time-series and attach noise to it
                ts1 = ts_obj1.generate(ts_length)
                ts2 = ts_obj2.generate(ts_length)

                if option.noise_snr1 > 0.0:
                    noise_level = calculate_noise_level(option.noise_snr1, ts_obj1, ts_length)
                    noise_obj = template['noise'](noise_level=noise_level)
                    ts1 = ts1 + noise_obj.generate(ts_length)
                if option.noise_snr2 > 0.0:
                    noise_level = calculate_noise_level(option.noise_snr2, ts_obj2, ts_length)
                    noise_obj = template['noise'](noise_level=noise_level)
                    ts2 = ts2 + noise_obj.generate(ts_length)

                #if ts_object has attribute transformation 
                if hasattr(ts_obj1, 'transformations'):
                    try:
                        for transformation_obj in ts_obj1.transformations:
                            ts1 = transformation_obj.transform(ts1)
                    except AnomalyOnePeakError:
                        print(f'ERROR: tid {tid}. Found one peak error. This is because the time series has only one peak. Increasing the length of the time series will usually solve this issue. Skipping this sample')
                        continue
                    except Exception as e:
                        raise Exception(f'Error in template: {template["tid"]}. Error: {e}')
                if hasattr(ts_obj2, 'transformations'):
                    try:
                        for transformation_obj in ts_obj2.transformations:
                            ts2 = transformation_obj.transform(ts2)
                    except AnomalyOnePeakError:
                        print(f'ERROR: tid {tid}. Found one peak error. This is because the time series has only one peak. Increasing the length of the time series will usually solve this issue. Skipping this sample')
                        continue
                    except Exception as e:
                        raise Exception(f'Error in template: {template["tid"]}. Error: {e}')
                
                #shuffle the options, replace answers with actual value if sampling involved in the 
                #corresponding parameters
                option_names = [execute_option(option.option_name) for option in template['options']]
                answer = option_names[i]
                option_names = [elem[1] for elem in option_names]
                if answer[0] == answer[1]:
                    answer = answer[0]
                else:
                    raise NotImplementedError('TwoTSOption parameter value answer not implemented')
                
                random.shuffle(option_names)

                qa_pairs.append({
                    'question': question,
                    'options': option_names,
                    'answer': answer,
                    'question_type': question_type,
                    'ts1': ts1,
                    'ts2': ts2,
                    'tid': tid,
                    'difficulty': difficulty,
                    'format_hint': template['format_hint'],
                    'relevant_concepts': template['relevant_concepts'],
                    'question_hint': template['question_hint']
                })

            elif isinstance(option, PairTSOption):
                ts_obj = get_pair_ts_obj_from_option(option)
                
                #get meta info  
                question = template['question']
                answer = option.option_name
                question_type = template['question_type']
                tid = template['tid']
                difficulty = template['difficulty']

                #generate time-series and attach noise to it
                ts1, ts2 = ts_obj.generate(ts_length)
                if option.noise_snr > 0.0:
                    noise_level = calculate_noise_level(option.noise_snr, ts_obj, ts_length)
                    noise_obj = template['noise'](noise_level=noise_level)
                    ts1 = ts1 + noise_obj.generate(ts_length)
                    ts2 = ts2 + noise_obj.generate(ts_length)
                
                #shuffle the options, replace answers with actual value if sampling involved in the 
                #corresponding parameters
                option_names = [execute_option(option.option_name) for option in template['options']]
                answer = option_names[i]
                option_names = [elem[1] for elem in option_names]
                if answer[0] == answer[1]:
                    answer = answer[0]
                else:
                    raise NotImplementedError('TwoTSOption answer not implemented')
                
                random.shuffle(option_names)

                qa_pairs.append({
                    'question': question,
                    'options': option_names,
                    'answer': answer,
                    'question_type': question_type,
                    'ts1': ts1,
                    'ts2': ts2,
                    'tid': tid,
                    'difficulty': difficulty,
                    'format_hint': template['format_hint'],
                    'relevant_concepts': template['relevant_concepts'],
                    'question_hint': template['question_hint']
                })
    return qa_pairs

def get_qa_dataset(num_questions_per_option, ts_length):
    qa_dataset = []

    for category, category_dict in TEMPLATE.items():
        for subcategory, subcategory_dict in category_dict.items():
            for template in subcategory_dict.values():
                try:
                    qa_pairs = get_qa_pairs(template, num_questions_per_option, ts_length)
                except Exception as e:
                    raise Exception(f'Error in template: {template["tid"]}. Category: {category}. Subcategory: {subcategory}, error: {e}')
                
                for qa_pair in qa_pairs:
                    qa_pair['category'] = category
                    qa_pair['subcategory'] = subcategory

                    #convert array to list 
                    if 'ts' in qa_pair:
                        qa_pair['ts'] = qa_pair['ts'].tolist()
                    else:
                        qa_pair['ts1'] = qa_pair['ts1'].tolist()
                        qa_pair['ts2'] = qa_pair['ts2'].tolist()
                    
                    for concept in qa_pair['relevant_concepts']:
                        assert concept in CONCEPTS, f'{concept} not in CONCEPTS'
                    qa_dataset.append(qa_pair)
    
    random.shuffle(qa_dataset)
    sample_id = 1
    for qa_sample in qa_dataset:
        qa_sample['id'] = sample_id
        sample_id += 1
    
    return qa_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate QA dataset')
    parser.add_argument('--num_questions_per_option', type=int, default=1, help='Number of questions per option')
    parser.add_argument('--ts_length', type=int, default=100, help='Length of the time series')
    parser.add_argument('--output_file', type=str, default='output/qa_dataset.json', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    args = parser.parse_args()

    seed_everything(args.seed)

    qa_dataset = get_qa_dataset(args.num_questions_per_option, args.ts_length)

    if not os.path.exists(args.output_file):
        os.makedirs(os.path.dirname(args.output_file))

    with open(args.output_file, 'w') as f:
        json.dump(qa_dataset, f, indent=4)
        
    print(f'Saved QA dataset with {len(qa_dataset)} samples to {args.output_file}')
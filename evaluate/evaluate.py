
import json 
from evaluation_utils import TimeSeriesQADataset
import logging 
from llm_api import LLM, LLM_HF, HF_MODELS, API_MODELS
from tqdm import tqdm 
import argparse 
import pdb 
import os 


logger = logging.getLogger('TS-QA-EVALUATION')

def evaluate_response(sample, response, mode='flexible'):
    if type(response) == float:
        response = str(response)
    answer = sample['answer']
    if type(answer) == float:
        answer = str(answer)
    #evaluate the response and store the results
    sample['response'] = response
    if mode == 'strict':
        sample['correct'] = answer.lower() in response.split('\n')[-1].lower()
    elif mode == 'flexible':
        answer_option_letter = chr(sample['options'].index(sample['answer']) + 65)
        flexible = f'{answer_option_letter}) {answer}'.lower() in response.lower()
        sample['correct'] = flexible

    if 'ts' in sample:
        del sample['ts']
    else:
        del sample['ts1']
        del sample['ts2']
    del sample['examples']

    return sample

def calculate_metric(results, metric='accuracy'):
    #obtain metric based on results 
    if metric == 'accuracy':
        correct = sum([1 for result in results if result['correct']])
        total = len(results)
        return correct / total
    else:
        raise NotImplementedError(f'Metric {metric} is not implemented')

def evaluate(args, keep_idx=None, cached_file=None):
    #define dataset (convert raw options into a qa format)
    with open(args.data_file_path, 'rb') as f:
        data = json.load(f)

    dataset = TimeSeriesQADataset(data, add_concepts=args.add_concepts, add_question_hint=args.add_question_hint, add_examples=args.add_examples)
    logger.info(f'Loaded dataset with {len(dataset)} samples')

    if args.ts_tokenizer_name == 'image':
        assert args.image_cache_dir is not None, 'Image cache directory cannot be None if ts_tokenizer_name is image'
        tokenizer_kwargs = {'image_cache_dir': args.image_cache_dir, 'mode': 'sample', 'model_name': args.model_name}
    else:
        tokenizer_kwargs = {'mode': 'sample'}

    #initialize LLM model
    if args.model_name in HF_MODELS:
        llm = LLM_HF(args.model_name,
                    ts_tokenizer_name=args.ts_tokenizer_name,
                    tokenizer_kwargs=tokenizer_kwargs,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=args.device)
    elif args.model_name in API_MODELS:
        llm = LLM(args.model_name,
                ts_tokenizer_name=args.ts_tokenizer_name,
                tokenizer_kwargs=tokenizer_kwargs,
                seed=args.seed,
                max_tokens=args.max_tokens,
                temperature=args.temperature)
    else:
        raise NotImplementedError(f'Model {args.model_name} is not implemented')
    
    logger.info(f'Initialized LLM model with model_name: {args.model_name}, ts_tokenizer_name: {args.ts_tokenizer_name}, seed: {args.seed}, max_tokens: {args.max_tokens}, temperature: {args.temperature}')
    
    #loop through dataset and query each question
    results = []

    for i in tqdm(range(len(dataset))):

        if keep_idx is not None:
            if i in keep_idx:
                results.append(cached_file[i])
                continue
        
        sample = dataset[i]
        query = sample['query']

        if 'ts' in sample:
            response = llm.query(query, sample['ts'], sample['examples'])
        else:
            assert 'ts1' in sample and 'ts2' in sample, 'ts1 and ts2 must be provided for two time series qa'
            response = llm.query(query, (sample['ts1'], sample['ts2']), sample['examples'])

        #evaluate the response and store the results
        result = evaluate_response(sample, response)
        results.append(result)

        # if i == 10:
        #     break

    #obtain metric based on results 
    metric = calculate_metric(results, metric=args.metric)
    results.append({args.metric: metric})
    logger.info(f'Obtained {args.metric} of {metric}')

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Time Series QA Model')
    parser.add_argument('--data_file_path', type=str, help='Path to the data file')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum number of tokens')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling')
    parser.add_argument('--metric', type=str, default='accuracy', help='Evaluation metric')
    parser.add_argument('--output_file_path', type=str, help='Path to the output file')
    parser.add_argument('--image_cache_dir', default=None, help='Path to the image cache directory, CANNOT be none if ts_tokenizer_name is image')
    parser.add_argument('--ts_tokenizer_name', type=str, help='Name of the time series tokenizer')
    parser.add_argument('--add_question_hint', action='store_true', help='Add question hint to the query')
    parser.add_argument('--add_concepts', action='store_true', help='Add concepts to the query')
    parser.add_argument('--add_examples', action='store_true', help='Add examples to the query')
    parser.add_argument('--keep_idx_path', type=str, required=False, help='Path to the keep idx file')
    parser.add_argument('--old_evaluation_dir', type=str, required=False, help='Path to the old evaluation directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for evaluation')

    args = parser.parse_args()

    output_file_path = args.output_file_path.replace('.json', f'_{args.model_name.replace('/', '-')}.json')
    output_file_path = output_file_path.replace('.json', f'_{args.ts_tokenizer_name}.json')
    if args.add_question_hint:
        output_file_path = output_file_path.replace('.json', '_hint.json')
    if args.add_concepts:
        output_file_path = output_file_path.replace('.json', '_concepts.json')
    if args.add_examples:
        output_file_path = output_file_path.replace('.json', '_examples.json')
    
    if args.keep_idx_path is not None:
        cache_file_path = os.path.join(args.old_evaluation_dir, output_file_path.split('/')[-1])
        assert os.path.exists(cache_file_path), f'Cache file {cache_file_path} does not exist'
        with open(cache_file_path, 'r') as f:
            cached_file = json.load(f)

        with open(args.keep_idx_path, 'r') as f:
            keep_idx = json.load(f)
    else:
        keep_idx = None
        cached_file = None

    results = evaluate(args, keep_idx=keep_idx, cached_file=cached_file)

    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)
        
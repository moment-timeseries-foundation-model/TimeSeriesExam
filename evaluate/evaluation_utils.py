import openai 
from openai import OpenAI
import anthropic
from anthropic import Anthropic
import google.generativeai as Gemini
import torch

import time 
from dataclasses import dataclass 
import numpy as np 
from typing import Union
import os 
import matplotlib.pyplot as plt 
import base64
import pdb 
import sys, os
from IPython.display import Image
from IPython.core.display import HTML
from PIL import Image as PILImage
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #TODO: this will be removed once we do setup tools

from torch.utils.data import Dataset, DataLoader
from concepts import CONCEPTS

@dataclass 
class PairTS:
    ts1: Union[list, np.array, str]
    ts2: Union[list, np.array, str]

@dataclass 
class SingleTS:
    ts: Union[list, np.array, str]


class TimeSeriesQADataset(Dataset):
    def __init__(self, data, add_concepts, add_question_hint, add_examples, max_concepts=3):
        self.data = data
        self.add_concepts = add_concepts
        self.add_question_hint = add_question_hint
        self.add_examples = add_examples
        self.max_concepts = max_concepts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample['question']
        options = sample['options']
        sample['examples'] = None

        #add letter to each option and join them with \n 
        options_string = '\n'.join([f'{chr(65 + i)}) {option}' for i, option in enumerate(options)])
        prompt = f"""{question}
        
        Choose From Following Options: 
        
        {options_string}\n"""

        if self.add_concepts:
            concepts_strings = [] 
            concepts_examples = []
            for idx, concept in enumerate(sample['relevant_concepts'][:self.max_concepts]):
                concept_name, concept_description = CONCEPTS[concept].concept_name, CONCEPTS[concept].concept_description

                if self.add_examples:
                    concepts_strings.append(f'{concept_name}: {concept_description}. {CONCEPTS[concept].concept_example_string}, marked as example {idx+1}.')
                    concepts_examples.append(CONCEPTS[concept].concept_example)
                else:
                    concepts_strings.append(f'{concept_name}: {concept_description}.')

            concepts_string = '\n'.join(concepts_strings)
            if self.add_examples:
                sample['examples'] = concepts_examples
            prompt += f"""Here are some relevant concepts: 
            {concepts_string}\n"""

        if self.add_question_hint:
            prompt += f"""Here is a hint that might help you: {sample['question_hint']}."""

    
        prompt += f"""{sample['format_hint']}. 

        Here is an example question answer pair to help you understand the format better:

        EXAMPLE QUESTION: 
        
        What is the most likely autocorrelation at lag 1 for the given time series?\n        \n        Choose From Following Options: \n        \n        A) High positive autocorrelation\nB) No autocorrelation\nC) Negative autocorrelation\nNow, answer the question.

        EXAMPLE RESPONSE: 
        
        Based on the given time series, the data points appear to fluctuate randomly around the mean with no clear pattern of persistence or trend. This suggests that the time series does not exhibit a strong relationship between consecutive data points.\n\nGiven the options:\n\nA) High positive autocorrelation\nB) No autocorrelation\nC) Negative autocorrelation\n\nThe most likely autocorrelation at lag 1 for the given time series is:\n\nB) No autocorrelation
        
        Now, answer the given question and also explain your thought process: """

        sample['query'] = prompt

        return sample

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def encode_image_gemini(image_path):
    return Image(image_path)
  
def plain_text_tokenizer(timeseries, kwargs):
    if type(timeseries) == tuple:
        timeseries_str1 = ",".join([str(round(x, 1)) for x in timeseries[0]])
        timeseries_str2 = ",".join([str(round(x, 1)) for x in timeseries[1]])
        return PairTS(ts1=timeseries_str1, ts2=timeseries_str2)
    elif type(timeseries) == list:
        try:
            timeseries_str = ",".join([str(round(x, 2)) for x in timeseries])
        except Exception as e:
            print(e)
            pdb.set_trace()
        return SingleTS(ts=timeseries_str)
    else:
        raise ValueError(f'Time Series Type {type(timeseries)} not recognized for plain text tokenizer')
    
def llmtime_tokenizer(text):
    raise NotImplementedError('llmtime_tokenizer method is not implemented')

def image_tokenizer(timeseries, kwargs):
    image_cache_dir = kwargs['image_cache_dir']
    mode = kwargs.get('mode', None)
    model_name = kwargs.get('model_name', None)
    if mode == 'example':
        example_idx = kwargs.get('example_idx', None)
    image_files = os.listdir(image_cache_dir)
    try:
        image_files.remove('.DS_Store')
    except:
        pass
    image_files = [int(file.split('.')[0]) for file in image_files]
    last_index = max(image_files) if len(image_files) > 0 else 0

    if type(timeseries) == tuple:
        img1_path = f'{image_cache_dir}/{last_index + 1}.jpg'
        img2_path = f'{image_cache_dir}/{last_index + 2}.jpg'
        #plot and save the two time series in .png 
        plt.plot(timeseries[0])
        if mode == 'example':
            plt.title(f'Example {example_idx} Time Series 1')
        else:
            plt.title('Time Series 1')
        plt.savefig(img1_path)
        plt.close()

        plt.plot(timeseries[1])
        if mode == 'example':
            plt.title(f'Example {example_idx} Time Series 2')
        else:
            plt.title('Time Series 2')
        plt.savefig(img2_path)
        plt.close()

        if 'gemini' in model_name.lower():
            return PairTS(ts1=encode_image_gemini(img1_path), ts2=encode_image_gemini(img2_path))
        elif 'phi' in model_name.lower() or 'minicpm' in model_name.lower():
            return PairTS(ts1=PILImage.open(img1_path), ts2=PILImage.open(img2_path))
        else:
            return PairTS(ts1=encode_image(img1_path), ts2=encode_image(img2_path))
    
    elif type(timeseries) == list:
        #plot and save the single time series in .png 
        img_path = f'{image_cache_dir}/{last_index + 1}.jpg'
        plt.plot(timeseries)
        if mode == 'example':
            plt.title(f'Example {example_idx} Time Series')
        else:
            plt.title('Time Series')
        plt.savefig(img_path)
        plt.close()
        if 'gemini' in model_name.lower():
            return SingleTS(ts=encode_image_gemini(img_path))
        elif 'phi' in model_name.lower() or 'minicpm' in model_name.lower():
            return SingleTS(ts=PILImage.open(img_path))
        else:
            return SingleTS(ts=encode_image(img_path))

def get_dummy_dataloader():
    class DummyDataset(Dataset):
        def __init__(self, num_samples=1000, input_size=10):
            # Create random data and random labels (for classification)
            self.data = torch.randn(num_samples, input_size)
            self.labels = torch.randint(0, 2, (num_samples,))  # Binary labels (0 or 1)
        
        def __len__(self):
            # Return the total number of samples
            return len(self.data)
        
        def __getitem__(self, idx):
            # Return a single sample (data and label)
            return self.data[idx], self.labels[idx]

    # Instantiate the dummy dataset
    dummy_dataset = DummyDataset(num_samples=1000, input_size=10)

    # Create a DataLoader
    dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=True)

    return dummy_loader

############################################model query functions################################################################
def query_gpt_4(client, model, messages, seed, max_gen=512, temperature=0.0, debug=False):
    # model_max = self.model_max
    # messages = self.shrink_msg(messages, shrink_idx, model_max-max_gen)
    while True:
        try:
            if isinstance(client, OpenAI):
                #print("client is present")
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_gen,
                    seed=seed,
                )
                return completion.choices[0].message.content
        except Exception as e:
            if debug:
                raise e
            elif (
                isinstance(e, openai.RateLimitError)
                or isinstance(e, openai.APIStatusError)
                or isinstance(e, openai.APITimeoutError)
                or isinstance(e, openai.APIConnectionError)
                or isinstance(e, openai.InternalServerError)
            ):
                time.sleep(30)
                print(e)

def chat_template_gpt_4(query, tokenized_ts, tokenization_type, examples=None):
    if tokenization_type == 'image':
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                base_message = "You are given one time series plotted in the image. Some examples of the relevant concepts are also plotted. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given one time series plotted in the image. Answer the following question based on the time series. Question: \n"
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": base_message + query},
                                    {"type": "image_url",
                                    "image_url": {"url": f'data:image/jpeg;base64,{tokenized_ts.ts}'}, 
                                    },
                                    ],
                        }
                    ]
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        messages[0]['content'].append({"type": "image_url",
                                                    "image_url": {"url": f'data:image/jpeg;base64,{example.ts}'}, 
                                                    })
                    else:
                        messages[0]['content'].append({"type": "image_url",
                                                        "image_url": {"url": f'data:image/jpeg;base64,{example.ts1}'}, 
                                                        })
                        messages[0]['content'].append({"type": "image_url",
                                                        "image_url": {"url": f'data:image/jpeg;base64,{example.ts2}'}, 
                                                        })
        elif isinstance(tokenized_ts, PairTS):
            if examples:
                base_message = "You are given two time series plotted in two images. Some examples of the relevant concepts are also plotted. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given two time series plotted in two images. Answer the following question based on the time series. Question: \n"
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": query},
                                    {"type": "image_url",
                                    "image_url": {"url": f'data:image/jpeg;base64,{tokenized_ts.ts1}'}, 
                                    },
                                    {"type": "image_url",
                                    "image_url": {"url": f'data:image/jpeg;base64,{tokenized_ts.ts2}'}, 
                                    },
                                    ],
                        }
                    ]
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        messages[0]['content'].append({"type": "image_url",
                                                    "image_url": {"url": f'data:image/jpeg;base64,{example.ts}'}, 
                                                    })
                    else:
                        messages[0]['content'].append({"type": "image_url",
                                                        "image_url": {"url": f'data:image/jpeg;base64,{example.ts1}'}, 
                                                        })
                        messages[0]['content'].append({"type": "image_url",
                                                        "image_url": {"url": f'data:image/jpeg;base64,{example.ts2}'}, 
                                                        })
        else:
            raise ValueError('Tokenized Time Series Type not recognized')
        
    elif tokenization_type == 'text':
        if examples:
            example_str = []
            for idx, example in enumerate(examples):
                if isinstance(example, SingleTS):
                    example_str.append(f'Example {idx+1} time series: {example.ts}')
                else:
                    example_str.append(f'Example {idx+1} time series1: {example.ts1}\nExample {idx+1} time series2: {example.ts2}')
            example_str = '\n'.join(example_str)
        else:
            example_str = ''
                    
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                base_message = "You are given one time series, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series: \n{timeseries}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts, example_str=example_str)
            else:
                base_message = "You are given one time series, where each step is separated by a comma. Timeseries: \n{timeseries}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts)
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": base_message + query},
                                    ],
                        }
                    ]
        elif isinstance(tokenized_ts, PairTS):
            if examples:
                base_message = "You are given two time series here, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series1: \n{timeseries1}\nTime series2: \n{timeseries2}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2, example_str=example_str)
            else:
                base_message = "You are given two time series here, where each step is separated by a comma. Timeseries1: \n{timeseries1}\nTimeseries2: \n{timeseries2}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2)
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": base_message + query},
                                    ],
                        }
                    ]
        else:
            raise ValueError('Tokenized Time Series Type not recognized')

    return messages

def query_claude(client, model, messages, seed, max_gen=512, temperature=0.0, debug=False):
    while True:
        try:
            if isinstance(client, Anthropic):
                #print("client is present")
                completion = client.messages.create(
                    model=model,
                    max_tokens=max_gen,
                    messages=messages,
                    temperature=temperature,
                    extra_headers={}
                )
                return completion.content[0].text
        except Exception as e:
            if debug:
                raise e
            elif (
                isinstance(e, anthropic.RateLimitError)
                or isinstance(e, anthropic.APIStatusError)
                or isinstance(e, anthropic.APITimeoutError)
                or isinstance(e, anthropic.APIConnectionError)
                or isinstance(e, anthropic.InternalServerError)
            ):
                time.sleep(30)
                print(e)

def chat_template_claude(query, tokenized_ts, tokenization_type, examples=None):
    if tokenization_type == 'image':
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                base_message = "You are given one time series plotted in the image. Some examples of the relevant concepts are also plotted. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given one time series plotted in the image. Answer the following question based on the time series. Question: \n"
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": base_message + query},
                                    {"type": "image",
                                    "source": {"type":'base64', "media_type":"image/jpeg", "data": tokenized_ts.ts}, 
                                    },
                                    ],
                        }
                    ]
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        messages[0]['content'].append({"type": "image",
                                                    "source": {"type":'base64', "media_type":"image/jpeg", "data": example.ts}, 
                                                    })
                    else:
                        messages[0]['content'].append({"type": "image",
                                                        "source": {"type":'base64', "media_type":"image/jpeg", "data": example.ts1}, 
                                                        })
                        messages[0]['content'].append({"type": "image",
                                                        "source": {"type":'base64', "media_type":"image/jpeg", "data": example.ts2}, 
                                                        })
        elif isinstance(tokenized_ts, PairTS):
            if examples:
                base_message = "You are given two time series plotted in two images. Some examples of the relevant concepts are also plotted. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given two time series plotted in two images. Answer the following question based on the time series. Question: \n"
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": query},
                                    {"type": "image",
                                    "source": {"type":'base64', "media_type":"image/jpeg", "data": tokenized_ts.ts1}, 
                                    },
                                    {"type": "image",
                                    "source": {"type":'base64', "media_type":"image/jpeg", "data": tokenized_ts.ts2}, 
                                    },
                                    ],
                        }
                    ]
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        messages[0]['content'].append({"type": "image",
                                                    "source": {"type":'base64', "media_type":"image/jpeg", "data": example.ts}, 
                                                    })
                    else:
                        messages[0]['content'].append({"type": "image",
                                                        "source": {"type":'base64', "media_type":"image/jpeg", "data": example.ts1}, 
                                                        })
                        messages[0]['content'].append({"type": "image",
                                                        "source": {"type":'base64', "media_type":"image/jpeg", "data": example.ts2}, 
                                                        })
        else:
            raise ValueError('Tokenized Time Series Type not recognized')
        
    elif tokenization_type == 'text':
        if examples:
            example_str = []
            for idx, example in enumerate(examples):
                if isinstance(example, SingleTS):
                    example_str.append(f'Example {idx+1} time series: {example.ts}')
                else:
                    example_str.append(f'Example {idx+1} time series1: {example.ts1}\nExample {idx+1} time series2: {example.ts2}')
            example_str = '\n'.join(example_str)
        else:
            example_str = ''
                    
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                base_message = "You are given one time series, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series: \n{timeseries}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts, example_str=example_str)
            else:
                base_message = "You are given one time series, where each step is separated by a comma. Timeseries: \n{timeseries}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts)
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": base_message + query},
                                    ],
                        }
                    ]
        elif isinstance(tokenized_ts, PairTS):
            if examples:
                base_message = "You are given two time series here, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series1: \n{timeseries1}\nTime series2: \n{timeseries2}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2, example_str=example_str)
            else:
                base_message = "You are given two time series here, where each step is separated by a comma. Timeseries1: \n{timeseries1}\nTimeseries2: \n{timeseries2}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2)
            messages=[
                        {
                        "role": "user",
                        "content": [
                                    {"type": "text", "text": base_message + query},
                                    ],
                        }
                    ]
        else:
            raise ValueError('Tokenized Time Series Type not recognized')

    return messages

def query_gemini(client, model, messages, seed, max_gen=512, temperature=0.0, debug=False):
    while True:
        try:
            response = client.generate_content(
                messages,
                generation_config=Gemini.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=max_gen,
                    temperature=temperature,
                ),
            )
            return response.text
        except Exception as e:
            if debug:
                raise e
            else:
                print(f"An error occurred: {e}")
                time.sleep(30)

def chat_template_gemini(query, tokenized_ts, tokenization_type, examples=None):
    if tokenization_type == 'image':
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                base_message = "You are given one time series plotted in the image. Some examples of the relevant concepts are also plotted. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given one time series plotted in the image. Answer the following question based on the time series. Question: \n"
            messages=[base_message + query,
                    tokenized_ts.ts]    
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        messages.append(example.ts)
                    else:
                        messages.append(example.ts1)
                        messages.append(example.ts2)
        elif isinstance(tokenized_ts, PairTS):
            if examples:
                base_message = "You are given two time series plotted in two images. Some examples of the relevant concepts are also plotted. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given two time series plotted in two images. Answer the following question based on the time series. Question: \n"
            messages=[query,
                        tokenized_ts.ts1,
                        tokenized_ts.ts2]
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        messages.append(example.ts)
                    else:
                        messages.append(example.ts1)
                        messages.append(example.ts2)
        else:
            raise ValueError('Tokenized Time Series Type not recognized')
        
    elif tokenization_type == 'text':
        if examples:
            example_str = []
            for idx, example in enumerate(examples):
                if isinstance(example, SingleTS):
                    example_str.append(f'Example {idx+1} time series: {example.ts}')
                else:
                    example_str.append(f'Example {idx+1} time series1: {example.ts1}\nExample {idx+1} time series2: {example.ts2}')
            example_str = '\n'.join(example_str)
        else:
            example_str = ''
                    
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                base_message = "You are given one time series, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series: \n{timeseries}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts, example_str=example_str)
            else:
                base_message = "You are given one time series, where each step is separated by a comma. Timeseries: \n{timeseries}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts)
            messages=[base_message + query]

        elif isinstance(tokenized_ts, PairTS):
            if examples:
                base_message = "You are given two time series here, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series1: \n{timeseries1}\nTime series2: \n{timeseries2}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2, example_str=example_str)
            else:
                base_message = "You are given two time series here, where each step is separated by a comma. Timeseries1: \n{timeseries1}\nTimeseries2: \n{timeseries2}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2)
            messages=[base_message + query]

        else:
            raise ValueError('Tokenized Time Series Type not recognized')

    return messages

def query_phi(client, model, inputs, seed, processor, max_gen=512, temperature=0.0, debug=False):
    inputs = inputs.to(client.device)
    
    generation_args = { 
                        "max_new_tokens": max_gen, 
                        "temperature": temperature, 
                        'do_sample': False,
                        } 

    generate_ids = client.generate(**inputs, 
                        eos_token_id=processor.eos_token_id, 
                        **generation_args
                        )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0] 

    return response

def chat_template_phi(query, tokenized_ts, tokenization_type, processor, examples=None):
    if tokenization_type == 'image':
        raise ValueError('Image tokenization not supported for phi')

    elif tokenization_type == 'text':
        if examples:
            example_str = []
            for idx, example in enumerate(examples):
                if isinstance(example, SingleTS):
                    example_str.append(f'Example {idx+1} time series: {example.ts}')
                else:
                    example_str.append(f'Example {idx+1} time series1: {example.ts1}\nExample {idx+1} time series2: {example.ts2}')
            example_str = '\n'.join(example_str)
        else:
            example_str = ''
                    
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                base_message = "You are given one time series, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series: \n{timeseries}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts, example_str=example_str)
            else:
                base_message = "You are given one time series, where each step is separated by a comma. Timeseries: \n{timeseries}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries=tokenized_ts.ts)
            messages=[
                        {
                        "role": "user",
                        "content": base_message + query
                        }
                    ]
        elif isinstance(tokenized_ts, PairTS):
            if examples:
                base_message = "You are given two time series here, where each step is separated by a comma. Some examples of the relevant concepts mentioned below are also provided.\nConcept example time series:\n{example_str}\nTime series1: \n{timeseries1}\nTime series2: \n{timeseries2}\n\nNow, answer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2, example_str=example_str)
            else:
                base_message = "You are given two time series here, where each step is separated by a comma. Timeseries1: \n{timeseries1}\nTimeseries2: \n{timeseries2}\nAnswer the following question based on the time series. In your analysis, try not to repeat large chunk of values in the time series to save space. Question: \n"
                base_message = base_message.format(timeseries1=tokenized_ts.ts1, timeseries2=tokenized_ts.ts2)
            messages=[
                        {
                        "role": "user",
                        "content": base_message + query
                        }
                    ]
        else:
            raise ValueError('Tokenized Time Series Type not recognized')

        prompts = processor.apply_chat_template(
                                                messages, 
                                                tokenize=False, 
                                                add_generation_prompt=True
                                                )
        inputs = processor(prompts, return_tensors="pt")

    return inputs

def query_cpm(client, model, inputs, seed, processor, max_gen=512, temperature=0.0, debug=False):
    
    generation_args = { 
                        "max_new_tokens": max_gen, 
                        "temperature": temperature, 
                        "do_sample": False
                        } 

    response = client.chat(image=None, msgs=inputs, tokenizer=processor, **generation_args)

    return response

def chat_template_cpm(query, tokenized_ts, tokenization_type, processor, examples=None):
    if tokenization_type == 'image':
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                num_examples = sum([1 if isinstance(example, SingleTS) else 2 for example in examples])
                placeholder = '\n'.join([f"<image_{j}>" for j in range(2, num_examples+2)])
                base_message = f"You are given one time series plotted in the image <image_1>\n. Some examples of the relevant concepts are also plotted {placeholder}. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given one time series plotted in the image <image_1>\n. Answer the following question based on the time series. Question: \n"  
            
            images = [tokenized_ts.ts] 
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        images.append(example.ts)
                    else:
                        images.append(example.ts1)
                        images.append(example.ts2)

        elif isinstance(tokenized_ts, PairTS):
            if examples:
                num_examples = sum([1 if isinstance(example, SingleTS) else 2 for example in examples])
                placeholder = '\n'.join([f"<image_{j}>" for j in range(3, num_examples+3)])
                base_message = f"You are given two time series plotted in two images <image_1>\n <image_2>\n. Some examples of the relevant concepts are also plotted {placeholder}. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given two time series plotted in two images <image_1>\n <image_2>\n. Answer the following question based on the time series. Question: \n"
            
            images = [tokenized_ts.ts1, tokenized_ts.ts2] 
            if examples:
                for example in examples:
                    if isinstance(example, SingleTS):
                        images.append(example.ts)
                    else:
                        images.append(example.ts1)
                        images.append(example.ts2)
        
        inputs = [{'role': 'user', 'content': images + [base_message + query]}]

    elif tokenization_type == 'text':
        raise ValueError('Text tokenization not supported for mini cpm')
    return inputs

def query_phi_v(client, model, inputs, seed, processor, max_gen=512, temperature=0.0, debug=False):
    inputs = inputs.to(client.device)
    
    generation_args = { 
                        "max_new_tokens": max_gen, 
                        "temperature": temperature, 
                        'do_sample': False,
                        } 
    with torch.no_grad():
        generate_ids = client.generate(**inputs, 
                            eos_token_id=processor.tokenizer.eos_token_id, 
                            **generation_args
                            )

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)[0] 

    return response

def chat_template_phi_v(query, tokenized_ts, tokenization_type, processor, examples=None):
    if tokenization_type == 'image':
        if isinstance(tokenized_ts, SingleTS):
            if examples:
                num_examples = sum([1 if isinstance(example, SingleTS) else 2 for example in examples])
                placeholder = '\n'.join([f"<|image_{j}|>" for j in range(2, num_examples+2)])
                base_message = f"You are given one time series plotted in the image <|image_1|>\n. Some examples of the relevant concepts are also plotted {placeholder}. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given one time series plotted in the image <|image_1|>\n. Answer the following question based on the time series. Question: \n"  
            
            messages = [
                        {"role": "user", "content": base_message + query},
                        ]
            images = [tokenized_ts.ts] 
            for example in examples:
                if isinstance(example, SingleTS):
                    images.append(example.ts)
                else:
                    images.append(example.ts1)
                    images.append(example.ts2)

        elif isinstance(tokenized_ts, PairTS):
            if examples:
                num_examples = sum([1 if isinstance(example, SingleTS) else 2 for example in examples])
                placeholder = '\n'.join([f"<|image_{j}|>" for j in range(3, num_examples+3)])
                base_message = f"You are given two time series plotted in two images <|image_1|>\n <|image_2|>\n. Some examples of the relevant concepts are also plotted {placeholder}. Answer the following question based on the time series. Question: \n"
            else:
                base_message = "You are given two time series plotted in two images <|image_1|>\n <|image_2|>\n. Answer the following question based on the time series. Question: \n"
            
            messages = [
                        {"role": "user", "content": base_message + query},
                        ]
            images = [tokenized_ts.ts1, tokenized_ts.ts2] 
            for example in examples:
                if isinstance(example, SingleTS):
                    images.append(example.ts)
                else:
                    images.append(example.ts1)
                    images.append(example.ts2)
        else:
            raise ValueError('Tokenized Time Series Type not recognized')
    
        prompt = processor.tokenizer.apply_chat_template(
                                                        messages, 
                                                        tokenize=False, 
                                                        add_generation_prompt=True
                                                        )
        
        inputs = processor(prompt, images, return_tensors="pt")

    elif tokenization_type == 'text':
        raise ValueError('Text tokenization not supported for phi_v')

    return inputs

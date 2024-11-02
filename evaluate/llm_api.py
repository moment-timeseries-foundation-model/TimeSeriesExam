from typing import Union 
from enum import StrEnum 
import os 
import pdb 
from evaluation_utils import (query_gpt_4, 
                                    OpenAI, 
                                    chat_template_gpt_4,
                                    query_claude,
                                    Anthropic,
                                    chat_template_claude,
                                    query_gemini,
                                    Gemini,
                                    query_cpm,
                                    query_phi,
                                    query_phi_v,
                                    chat_template_cpm,
                                    chat_template_phi,
                                    chat_template_phi_v,
                                    chat_template_gemini,
                                    plain_text_tokenizer,
                                    llmtime_tokenizer,
                                    image_tokenizer,
                                    get_dummy_dataloader)
from transformers import (AutoModelForCausalLM,
                          AutoProcessor,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          AutoModel)
import torch

class ModelType(StrEnum):
    GPT = 'GPT'
    Claude = 'Claude'
    Gemini = 'Gemini'
    MiniCPM = 'MiniCPM'
    Phi = 'Phi'
    PhiV = 'PhiV'

MODEL_TYPE = {
    "gpt-4o-mini": ModelType.GPT,
    "gpt-4o": ModelType.GPT,
    "claude-3-5-sonnet-20240620": ModelType.Claude,
    "gemini-1.5-pro": ModelType.Gemini,
    'openbmb/MiniCPM-V-2_6': ModelType.MiniCPM,
    "microsoft/Phi-3.5-vision-instruct": ModelType.PhiV,
    "microsoft/Phi-3.5-mini-instruct": ModelType.Phi
}

HF_MODELS = ["microsoft/Phi-3.5-vision-instruct", "microsoft/Phi-3.5-mini-instruct", 'openbmb/MiniCPM-V-2_6']
API_MODELS = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20240620", "gemini-1.5-pro"]

QUERY_FUNC = {
    ModelType.GPT: query_gpt_4,
    ModelType.Claude: query_claude,
    ModelType.Gemini: query_gemini,
    ModelType.MiniCPM: query_cpm,
    ModelType.Phi: query_phi,
    ModelType.PhiV: query_phi_v
}

FORMAT_FUNC = {
    ModelType.GPT: chat_template_gpt_4,
    ModelType.Claude: chat_template_claude,
    ModelType.Gemini: chat_template_gemini,
    ModelType.MiniCPM: chat_template_cpm,
    ModelType.Phi: chat_template_phi,
    ModelType.PhiV: chat_template_phi_v
}

CLIENT = {
    ModelType.GPT: OpenAI,
    ModelType.Claude: Anthropic,
}

API_KEY_ENVIRON = {
    ModelType.GPT: 'OPENAI_API_KEY',
    ModelType.Claude: 'ANTHROPIC_API_KEY',
    ModelType.Gemini: 'GOOGLE_API_KEY'
}

API_KEY_PATHS = {
    ModelType.GPT: '.openai.key',
    ModelType.Claude: '.anthropic.key',
    ModelType.Gemini: '.google.key'
}


def get_api_key(model_type: ModelType) -> None:
    
    api_key_path = API_KEY_PATHS.get(model_type)
    if api_key_path is not None:
        local_api_key_path = os.path.join(os.path.expanduser("~"), api_key_path)

    if os.path.exists(local_api_key_path):
        with open(local_api_key_path, "r") as f:
            lines = f.read().splitlines()
            api_key = lines[0].strip()
    else:
        api_key = os.environ.get(API_KEY_ENVIRON.get(model_type, None), None)
        
        if api_key is None:
            raise ValueError(f"API key not found for {model_type}")

    return api_key

 
def init_client(model_type: ModelType):
    api_key = get_api_key(model_type)
    if model_type == ModelType.Gemini:
        Gemini.configure(api_key=api_key)
        client = Gemini.GenerativeModel(model_name="gemini-1.5-pro")
        return client
    try:
        client = CLIENT.get(model_type)(api_key=api_key)
        return client
    except:
        raise ValueError("Model type not supported")

def init_client_hf(model_type: ModelType, device:str='auto'):
    if model_type == ModelType.PhiV:
        client = AutoModelForCausalLM.from_pretrained(
                                                    "microsoft/Phi-3.5-vision-instruct", 
                                                    device_map=device, 
                                                    trust_remote_code=True, 
                                                    torch_dtype="auto", 
                                                    )
        return client
    elif model_type == ModelType.Phi:
        bnb_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                        )

        client = AutoModelForCausalLM.from_pretrained(
                                                    "microsoft/Phi-3.5-mini-instruct", 
                                                    trust_remote_code=True, 
                                                    quantization_config=bnb_config,
                                                    torch_dtype=torch.bfloat16,
                                                    device_map={'':torch.cuda.current_device()}
                                                    )
        return client
    elif model_type == ModelType.MiniCPM:
        client = AutoModel.from_pretrained(
                                            'openbmb/MiniCPM-V-2_6', 
                                            device_map=device, 
                                            trust_remote_code=True, 
                                            torch_dtype="auto", 
                                            ).eval()
        return client
    else:
        raise ValueError(f"Model {model_type} type not supported")
    
def init_ts_tokenizer(ts_tokenizer_name:str):
    '''
    initialize the time series tokenizer 
    '''
    if ts_tokenizer_name == 'plain_text':
        return plain_text_tokenizer
    elif ts_tokenizer_name == 'llmtime':
        return llmtime_tokenizer
    elif ts_tokenizer_name == 'image':
        return image_tokenizer
    else:
        raise NotImplementedError(f'{ts_tokenizer_name} is not implemented, choose from [plain_text, llmtime, image]')
    
class LLM: 
    def __init__(self,
                 model_name:str,
                 ts_tokenizer_name:str,
                 tokenizer_kwargs:dict=None,
                 seed:int=42,
                 max_tokens:int=1024,
                 temperature:float=0.0,
                ):
        self.model_name = model_name
        self.model_type = MODEL_TYPE[self.model_name]
        self.client = init_client(self.model_type)
        self.seed = seed 
        self.query_func = QUERY_FUNC[self.model_type]
        self.ts_tokenizer_name = ts_tokenizer_name
        self.ts_tokenizer = init_ts_tokenizer(self.ts_tokenizer_name)
        self.tokenizer_kwargs = tokenizer_kwargs                                                       

        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def query(self, query:str, timeseries, examples=None) -> str: 
        '''
        timeseires: Union[list, tuple[list]] #one ts or two ts qa 
        '''
        message = self.format_query(query, timeseries, examples)
        return self.query_func(self.client, self.model_name,
                               messages=message,
                               seed=self.seed,
                               max_gen=self.max_tokens,
                               temperature=self.temperature)
    
    def format_query(self, query:str, timeseries, examples=None):
        '''
        query: the question we want to ask 
        timeseries: the time series we want to ask about

        apply chat template to the query and timeseries, this could involve 
        time series tokenization or convert to a diagram 
        '''

        tokenized_ts = self.ts_tokenizer(timeseries, self.tokenizer_kwargs)
        if examples is not None:
            tokenized_examples = []
            for idx, example in enumerate(examples):
                kwargs = self.tokenizer_kwargs.copy()
                kwargs['mode'] = 'example'
                kwargs['example_idx'] = idx + 1
                if len(example) == 1:
                    tokenized_example = self.ts_tokenizer(list(example[0]), kwargs)
                else:
                    assert len(example) == 2, 'Example must be a list of one or two time series'
                    tokenized_example = self.ts_tokenizer((list(example[0]), list(example[1])), kwargs)
                tokenized_examples.append(tokenized_example)
        else:
            tokenized_examples = None

        if self.ts_tokenizer_name == 'image':
            return FORMAT_FUNC[self.model_type](query, tokenized_ts, 'image', tokenized_examples)

        elif self.ts_tokenizer_name == 'llmtime' or self.ts_tokenizer_name == 'plain_text':
            return FORMAT_FUNC[self.model_type](query, tokenized_ts, 'text', tokenized_examples)

        else:
            raise NotImplementedError(f'{self.ts_tokenizer_name} is not implemented, choose from [plain_text, llmtime, image]')

class LLM_HF: 
    def __init__(self,
                 model_name:str,
                 ts_tokenizer_name:str,
                 tokenizer_kwargs:dict=None,
                 seed:int=42,
                 max_tokens:int=1024,
                 temperature:float=0.0,
                 device:str='auto'
                ):
        self.model_name = model_name
        self.model_type = MODEL_TYPE[self.model_name]
        self.client = init_client_hf(self.model_type, device)
        self.seed = seed 
        self.query_func = QUERY_FUNC[self.model_type]
        self.ts_tokenizer_name = ts_tokenizer_name
        self.ts_tokenizer = init_ts_tokenizer(self.ts_tokenizer_name)
        self.tokenizer_kwargs = tokenizer_kwargs  

        if self.model_type == ModelType.PhiV:
            self.processor = AutoProcessor.from_pretrained(self.model_name,
                                                           trust_remote_code=True)
        elif self.model_type == ModelType.MiniCPM:
            self.processor = AutoTokenizer.from_pretrained(self.model_name,
                                                            trust_remote_code=True)
        else:
            self.processor = AutoTokenizer.from_pretrained(self.model_name,
                                                           trust_remote_code=True)

        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def query(self, query:str, timeseries, examples=None) -> str: 
        '''
        timeseires: Union[list, tuple[list]] #one ts or two ts qa 
        '''
        inputs = self.format_query(query, timeseries, examples)
        return self.query_func(self.client, self.model_name,
                               inputs=inputs,
                               seed=self.seed,
                               processor=self.processor,
                               max_gen=self.max_tokens,
                               temperature=self.temperature)
    
    def format_query(self, query:str, timeseries, examples=None):
        '''
        query: the question we want to ask 
        timeseries: the time series we want to ask about

        apply chat template to the query and timeseries, this could involve 
        time series tokenization or convert to a diagram 
        '''

        tokenized_ts = self.ts_tokenizer(timeseries, self.tokenizer_kwargs)
        if examples is not None:
            tokenized_examples = []
            for idx, example in enumerate(examples):
                kwargs = self.tokenizer_kwargs.copy()
                kwargs['mode'] = 'example'
                kwargs['example_idx'] = idx + 1
                if len(example) == 1:
                    tokenized_example = self.ts_tokenizer(list(example[0]), kwargs)
                else:
                    assert len(example) == 2, 'Example must be a list of one or two time series'
                    tokenized_example = self.ts_tokenizer((list(example[0]), list(example[1])), kwargs)
                tokenized_examples.append(tokenized_example)
        else:
            tokenized_examples = None

        if self.ts_tokenizer_name == 'image':
            return FORMAT_FUNC[self.model_type](query, tokenized_ts, 'image', self.processor, tokenized_examples)

        elif self.ts_tokenizer_name == 'llmtime' or self.ts_tokenizer_name == 'plain_text':
            return FORMAT_FUNC[self.model_type](query, tokenized_ts, 'text', self.processor, tokenized_examples)

        else:
            raise NotImplementedError(f'{self.ts_tokenizer_name} is not implemented, choose from [plain_text, llmtime, image]')
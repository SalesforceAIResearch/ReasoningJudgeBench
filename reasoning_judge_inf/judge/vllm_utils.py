
import time
from openai import OpenAI
from vllm import SamplingParams
from datasets import Dataset
from transformers import AutoTokenizer as AT


MAX_CONNECTION_TRIES = 240 # number of times to try connecting judge and refiner endpoints
DELAY_TIME = 10 # sleep for 10s; 240 x 10 = 40 minutes before giving up

def wait_until_server_ready(base_url, api_key='sample-api-key', max_connection_tries=None, delay_time=None):
    if max_connection_tries is None:
        max_connection_tries = MAX_CONNECTION_TRIES
    if delay_time is None:
        delay_time = DELAY_TIME
    for i in range(max_connection_tries):
        try:
            if 'together' in base_url:
                return
            
            client = OpenAI(base_url=base_url, api_key=api_key)
            client.models.list()
            print(client.models.list())
            return
        except:
            print(f'Waiting for server to be ready... {i+1} / {max_connection_tries}')
            time.sleep(delay_time)
    raise ConnectionError(f'Cannot connect to server after {max_connection_tries} tries.')

class VllmEndpoint():
    def __init__(self, endpoint_config):    
        '''
        endpoint_config is dict with keys ["base_url", "api_key", "model_name"]
        '''
        self.base_url = endpoint_config['base_url']
        self.api_key = endpoint_config['api_key']

        wait_until_server_ready(self.base_url, api_key=self.api_key)
        self.model_name = self.get_model_name(endpoint_config["model_name"])
        self.tokenizer = AT.from_pretrained(self.model_name)
    
    def get_model_name(self, model_name):
        if model_name is not None:
            return model_name

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        model_list = list(client.models.list())
        assert len(model_list) == 1, 'The vllm server is serving multiple models? This doesn\'t seem right...'
        return model_list[0].id

    def generate(self, messages: list, sp: SamplingParams, return_comp_tokens: bool =False, completion_gen: bool = False):
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        received = False
        cnt_try = 0
        comp_tokens = None
        while not received and cnt_try < 3:
            try:
                if not completion_gen:
                    response = client.chat.completions.create(
                        model=self.model_name, 
                        messages=messages, 
                        temperature=sp.temperature, 
                        max_tokens=sp.max_tokens, 
                        top_p=sp.top_p, 
                        n=sp.n,
                    )

                    texts = [r.message.content for r in response.choices]

                else:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompt = ''.join(prompt.split(self.tokenizer.eos_token)[:-1])
                    response = client.completions.create(
                        model=self.model_name,
                        prompt=prompt, 
                        temperature=sp.temperature, 
                        max_tokens=sp.max_tokens, 
                        top_p=sp.top_p, 
                        n=sp.n,
                    )
                    
                    texts = [r.text for r in response.choices]
                
                if return_comp_tokens:
                    comp_tokens = response.usage.completion_tokens
                received = True
            except Exception as e:
                texts = ["None"]
                cnt_try += 1

                print(f"API error; Retrying maximum {3 - cnt_try} times. Error: {e}")
                time.sleep(10)

        return texts, comp_tokens
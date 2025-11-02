from dataclasses import dataclass
from openai import OpenAI
from app.settings import SETTINGS
import gradio as gr
from agents import Agent, ModelSettings, function_tool,Runner, RunContextWrapper, InputGuardrail, GuardrailFunctionOutput, OutputGuardrail
import asyncio
import pandas as pd
import threading
import time





CLIENT = OpenAI()

@dataclass
class relevance_response():
    is_relevant: bool
    reason: str
    def __str__(self):
        return f'-Is it Relevant: \n     {self.is_relevant}\n-Reasoning:\n     {self.reason}'
    
@dataclass
class descriptive_response():
    is_descriptive: bool
    #is_informative: bool
    reason: str
    def __str__(self):
        #self.is_descriptive = self.is_informative
        return f'-Is it Descritive: \n     {self.is_descriptive}\n-Reasoning:\n     {self.reason}'

relevance_agent = Agent(
                        name="Relevance check",
                        instructions="Check if the input's overall topic is highly relevant to the field of fertility, medicine or embryology.",
                        output_type=relevance_response,
                        model='gpt-4.1-mini'
                        )

async def relevance_guardrail(ctx, agent, input_data):
    print('\n\n---------Guard----------\n\n')
    result = await Runner.run(relevance_agent, input_data)
    final_output = result.final_output_as(relevance_response)
    print(f'Relevant: {final_output.is_relevant}')
    print(f'Reason: {final_output.reason}')
    print('\n\n---------Guard End-------\n\n')
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_relevant,
    )

class ExtractAgent:
    name = "ExtractAgent"
    def __init__(self):
        self.client = OpenAI()
        self.agent_r = Agent(
                            name='ExtractAgent - Relevancy',
                            instructions='You are a student rushing an assignment.\n' \
                                         'What is the topic of this input in 10 words or less.',
                            model=SETTINGS.chat_model,
                            input_guardrails=[
                                InputGuardrail(guardrail_function=relevance_guardrail),
                            ],
                          )
        self.agent_d = Agent(
                            name='ExtractAgent - Descriptive',
                            instructions='You are a strict documentarian.\n' \
                                         'Does this excerpt alone' \
                                         'provide you with enough information ' \
                                         'to describe a topic involving ' \
                                         'medicine, fertility, or embryology?\n' \
                                         'if yes set is_descriptive to true.\n'
                                         'Answer in a concise fashion.',
                            model='gpt-5-nano',
                            output_type=descriptive_response
                          )
        #self.series = None
    # async def analysis_worker(self, indexes):
    #     for i in indexes:
    #         response = await Runner.run(starting_agent=self.agent_d,
    #                                 input=self.series[i]
    #                                 )
    #         final_output = response.final_output_as(descriptive_response)
    #         print(f'Input: {self.series[i]}\n{final_output}\n')
    #         self.series[i] = final_output.is_descriptive
    #     pass
    # async def run_analysis(self):
    #     print('\n\n---------Analysis----------\n\n')
    #     nd = len(input)
    #     self.series = pd.Series(range(nd))
    #     if(nd != 0):
    #         n = 8
    #         if(nd//n == 0):
    #             n = nd-1
    #         print((0,nd,nd//n,n))
    #         x = list(range(0,nd,nd//n))
    #         threads = []
    #         for i in range(0,n):
    #             if(i == n-1):
    #                 chunk = input[x[i]:]
    #             else:
    #                 chunk = input[x[i]:x[i+1]]
    #             threads.append(threading.Thread(
    #                                             target=self.analysis_worker,
    #                                             args=(chunk)
    #                                             )
    #                         )
    #             threads[i].start()
    #         for thread in threads:
    #             thread.join()
    #     print('\n\n---------Analysis End-------\n\n')

    async def summarize(self, input) -> list:
        response = await Runner.run(starting_agent=self.agent_r,
                                    input=input
                                    )
        
        #print(f'Final output:\n{response.final_output}')
        # contexts += response.citations
        return response.final_output

    def answer(self, input: str) -> str:
        return asyncio.run(self.summarize(input))
    
    async def run_analysis(self, input_):
        response = await Runner.run(
                                    starting_agent=self.agent_d,
                                    input=input_
                                   )
        final_output = response.final_output_as(descriptive_response)
        print(f'\nInput: {input_}\n{final_output}\n')
        return final_output.is_descriptive

    def analyze(self, input: str)->bool:
        return asyncio.run(self.run_analysis(input))
        
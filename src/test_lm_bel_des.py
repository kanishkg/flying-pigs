import argparse
import os
import time
import csv

import numpy as np
import openai
from crfm_llm import crfmLLM




default_template = """Does the following response to the question imply yes or no? Write your answer as yes or no.
Q: {names[0]} asked "{question}" and {names[1]} responded "{answer}"
So, {question}
A:"""


cot_template = """Does the following response to the question imply yes or no? Write your answer as yes or no. Think step by step before answering.
Here is how to answer questions:
Q: Some question
Thought: Let's think step by step to get the right answer.
Write down the thought and steps to arrive at an answer.
A: yes or no

Q: {names[0]} asked "{question}" and {names[1]} responded "{answer}", which means?
Thought:Let's think step by step to get the right answer."""

belief_desire_template = """Does the following response to the question imply yes or no? Write your answer as yes or no.
Here is how to answer questions:
Q: Some question
Thought:Let's think through the beliefs and desires of the responder.
Belief: Write down beliefs 
Desire: Write down desires
A: yes or no

Q: {names[0]} asked "{question}" and {names[1]} responded "{answer}", which means?
Thought:Let's think through the beliefs and desires of the responder.
Belief:"""


belief_template = """{names[0]} asked "{question}" and {names[1]} responded "{answer}"

What are three possible beliefs that {names[1]} has that would cause them to say this? 

Possible beliefs the responder could have include:
1)"""

desire_template = """{names[0]} asked "{question}" and {names[1]} responded "{answer}"

What are three possible desires that {names[1]} has that would cause them to say this? 

Possible desires the responder could have include:
1)"""

back_template = """Does the following response to the question imply yes or no? Write your answer as yes or no.
Q: {names[0]} asked "{question}"
Belief of {names[1]}: {belief}
Desire of {names[1]}: {desire}
{names[1]} responded: "{answer}"
"""

def prompt_crfm(prompt, llm):
    result = llm(prompt)
    return result

def get_args():
    parser = argparse.ArgumentParser(description="Run an implicature understanding run.")
    
    # prompt arguments
    parser.add_argument("--prompt", type=str, default="default")
    
    # language model arguments
    parser.add_argument("--model", type=str, default="code-davinci-002",
                        help='the model to use for the game')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=200)
    # TODO: add self consistent option if needed
    # parser.add_argument("--self_consistency", type=int, default=1)
    parser.add_argument("-n", "--num_questions", type=int, default=1)
    parser.add_argument("-o", "--offset", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="dev")
    parser.add_argument("--human_agreement", action="store_true")

    # output arguments
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--echo", action="store_true")
 
    return parser.parse_args()

def load_data(args):
    file_path = f"data/{args.dataset}_conversational_implicatures.csv"
    with open(file_path, "r") as f:
        data = list(csv.reader(f))[1:]

    print(f"Loaded {len(data)} examples from {file_path}")
    if args.human_agreement:
        with open("data/human_eval/human_eval - 1-150.csv", "r") as f:
            human_data = list(csv.reader(f))[1:]
        with open("data/human_eval/human_eval - 151-300.csv", "r") as f:
            human_data += list(csv.reader(f))[1:]
        with open("data/human_eval/human_eval - 301-450.csv", "r") as f:
            human_data += list(csv.reader(f))[1:]
        with open("data/human_eval/human_eval - 451-600.csv", "r") as f:
            human_data += list(csv.reader(f))[1:]
        print(f"Loaded {len(human_data)} human examples")
        assert len(data) == len(human_data), f"Data and human data have different lengths: {len(data)} and {len(human_data)}"
        for i, d in enumerate(human_data):
            annotations = [int(x == "Yes") for x in d[3:8]]
            human_label = int(sum(annotations) > 2)
            if human_label == 0:
                data[i][-1] = "no."
            elif human_label == 1:
                data[i][-1] = "yes."
    return data

def get_response_list(cur_prompt, llm=None):
    result = prompt_crfm(cur_prompt, llm)
    response_list = result.strip().splitlines()
    response_list = [response_list[0]] + [response.split(')', 1)[-1].strip() for response in response_list[1:]]
    return response_list
        
def get_echo_probs(prompt, llm, prefix="A"):
    result = llm.echo_prompt(llm, prompt=prompt)
    # for crfm
    tokens = result.tokens
    for i, token in enumerate(tokens):
        if token.text.strip() in prefix and tokens[i+1].text == ':':
            answer_token_idx = i+1
            break
    logprobsum = sum([t.logprob for t in tokens[answer_token_idx+1:]])
    # for openai
    # else:
    #     offsets = result['text_offset']
    #     start_offset = len(prompt)
    #     answer_offset = prompt.find('A:')
    #     answer_token_idx = next(x for x, val in enumerate(offsets) if val > answer_offset)
    #     logprobsum = sum(result['token_logprobs'][answer_token_idx - 1:])
    return np.exp(logprobsum)

def main():
    args = get_args()
    data = load_data(args)
    # use constant names for now
    names = ["Alice", "Bob"]
    api_key = os.getenv("OPENAI_API_KEY")
    if 'openai' in args.model:
        lm = crfmLLM(model_name=args.model, max_tokens=args.max_tokens, top_p=args.top_p, temperature=args.temperature)
    else:
        raise NotImplementedError(f"{args.model} not yet implemented")

    score = 0
    total = 0
    for n in range(args.offset, min(args.num_questions,len(data))):

        question, answer, target = data[n]
        if args.prompt == "default":
            template = default_template
        elif args.prompt == "cot":
            template = cot_template
        elif args.prompt == "bel-des-0":
            template = belief_desire_template
        elif args.prompt == "bel-des-back":
            template = [belief_template, desire_template, back_template]
        else:
            print(f"{question}\n{answer}\n{target}")
            raise NotImplementedError(f"{args.prompt} not yet implemented")
        not_target = "no." if target == "yes." else "yes."
        if args.prompt == "default":
            prompt = template.format(names=names, question=question, answer=answer)
            if args.echo:
                prompt_right = prompt+target
                prompt_wrong = prompt+not_target                           
                probs_right = get_echo_probs(prompt_right, lm)
                probs_wrong = get_echo_probs(prompt_wrong, lm)
                if probs_right > probs_wrong:
                    score += 1
            else:
                prompt_write = prompt+f"\nSo, {question}\nA:"
                ans = prompt_crfm(prompt_write, lm)
                if ans.strip().lower() in target:
                    score+=1
                elif ans.strip().lower() in not_target:
                    pass
                else:
                    print(f"Answer not in target or not_target: {question}\n {ans} \nTarget: {target}")                    # ask for input
                    inp = input("Enter eval answer(0/1): ")
                    c = int(inp)
                    score += c

        elif args.prompt == "cot":
            prompt = template.format(names=names, answer=answer, question=question)
            thought = prompt_crfm(prompt, lm)
            if args.echo:
                prompt_right = prompt+thought+f"\nSo, {question}\nA:"+target
                prompt_wrong = prompt+thought+f"\nSo, {question}\nA:"+not_target                           
                probs_right = get_echo_probs(prompt_right, lm)
                probs_wrong = get_echo_probs(prompt_wrong, lm)
                if probs_right > probs_wrong:
                    score += 1
            else:
                prompt_write = prompt+thought+f"\nSo, {question}\nA:"
                ans = prompt_crfm(prompt_write, lm)
                if ans.strip().lower() in target:
                    score +=1
                elif ans.strip().lower() in not_target:
                    pass
                else:
                    print(f"Answer not in target or not_target: {question}\n {ans} \nTarget: {target}")                    # ask for input
                    inp = input("Enter eval answer(0/1): ")
                    c = int(inp)
                    score += c
 

        elif args.prompt == "bel-des-0":
            prompt = template.format(names=names, answer=answer, question=question)
            bels = prompt_crfm(prompt, lm)
            prompt_bel = prompt+bels+"\nDesire:"
            des = prompt_crfm(prompt_bel, lm)
            prompt_des = prompt_bel+des
            if args.echo:
                prompt_right = prompt_des+f"\nSo, {question}\nA:"+target
                prompt_wrong = prompt_des+f"\nSo, {question}\nA:"+not_target                           
                probs_right = get_echo_probs(prompt_right, lm)
                probs_wrong = get_echo_probs(prompt_wrong, lm)
                if probs_right > probs_wrong:
                    score += 1
            else:
                prompt_write = prompt_des+f"\nSo, {question} Yes or No?\nA:"
                ans = prompt_crfm(prompt_write, lm)
                if ans.strip().lower() in target:
                    score+=1
                elif ans.strip().lower() in not_target:
                    pass
                else:
                    print(f"Answer not in target or not_target: {question}\n {ans} \nTarget: {target}")                    # ask for input
                    inp = input("Enter eval answer(0/1): ")
                    c = int(inp)
                    score += c
 
        elif args.prompt == "bel-des-back":
            belief_prompt = template[0].format(names=names, answer=answer, question=question)
            belief_list = get_response_list(belief_prompt, lm)
            desire_prompt = template[1].format(names=names, answer=answer, question=question)
            desire_list = get_response_list(desire_prompt, lm)
            highest_logprob = -np.inf
            for b in belief_list:
                for d in desire_list:
                    prompt = template[2].format(names=names, answer=answer, question=question, belief=b, desire=d)
                    resp_prob = get_echo_probs(prompt, lm, prefix="responded")
                    if resp_prob > highest_logprob:
                        highest_logprob = resp_prob
                        highest_prompt = prompt
            if args.echo:
                prompt_right = highest_prompt+f"\nSo, {question}\nA:"+target
                prompt_wrong = highest_prompt+f"\nSo, {question}\nA:"+not_target                           
                probs_right = get_echo_probs(prompt_right, lm)
                probs_wrong = get_echo_probs(prompt_wrong, lm)
                if probs_right > probs_wrong:
                    score += 1
            else:
                prompt_write = highest_prompt+f"\nSo, {question} Yes or No?\nA:"
                print(prompt_write)
                ans = prompt_crfm(prompt_write, lm)
                if ans.strip().lower() in target:
                    score+=1
                elif ans.strip().lower() in not_target:
                    pass
                else:
                    
                    print(f"Answer not in target or not_target: {question}\n {ans} \nTarget: {target}")
                    # ask for input
                    inp = input("Enter eval answer(0/1): ")
                    c = int(inp)
                    score += c
        total += 1
        print(f"Score: {score}/{total} = {score/total}")
    print(f"Final Score: {score}/{total} = {score/total}")

if __name__ == "__main__":
    main()
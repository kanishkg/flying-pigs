import argparse
import os
import time
import csv

import openai
from synchromesh import completion_engine, language_model
from synchromesh.synchromesh import predict_constrained




default_prompt = "Does the following response to the question imply yes or no? Write your answer as yes or no.\n"
scripted_prompt = """Answer the following questions about implicatures. What does Juan mean when he says his dialogue.

Q: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means?
A: Let's think about this step by step:
Esther said: “Can you come to my party on Friday?”
Esther's intent: Esther wants Juan to come to her party on Friday
Esther's belief: Esther believes Juan can come to her party on Friday
Juan said: “I have to work”
Juan's intent: Juan doesn't want to come to Esther's party on Friday
Juan's belief: Juan believes he has to work on Friday
Implicature: No, Juan cannot go to the party on Friday.
Answer:no~\n"""

thought_prompt = """Answer the following questions about implicatures.

Q: Esther asked “Can you come to my party on Friday?” and Juan responded “I have to work”, which means?

A: Let's think step by step:
Esther said: “Can you come to my party on Friday?”
So, she wants to know if Juan could come to her party on Friday. She currently does not know if Juan is free on Friday.
Juan responded: “I have to work”
So, Juan wants to tell Esther that he cannot come to the party on Friday. To communicate the reason for his absence, he says that he has to work.
To answer Esther's question, “Can you come to my party on Friday?” Juan is saying no.
Answer:no~
"""

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
    parser.add_argument("--max_tokens", type=int, default=300)
    # TODO: add self consistent option if needed
    # parser.add_argument("--self_consistency", type=int, default=1)
    parser.add_argument("-n", "--num_questions", type=int, default=1)
    parser.add_argument("-o", "--offset", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="dev")
    parser.add_argument("--human_agreement", action="store_true")

    # output arguments
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_synchromesh", action="store_true")
    parser.add_argument("--output_dir", type=str, default="out")
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


def main():
    args = get_args()
    data = load_data(args)
    prompt = ""
    # use constant names for now
    names = ["Alice", "Bob"]
    api_key = os.getenv("OPENAI_API_KEY")
    lm = language_model.OpenAIModel(model=args.model, api_key=api_key, prompt_template="",
                                                temperature=args.temperature, top_p=args.top_p, best_of=args.best_of)
    score = 0
    total = 0
    for n in range(args.offset, min(args.num_questions,len(data))):
        if args.prompt == "default":
            prompt = default_prompt
        elif args.prompt == "scripted":
            prompt = scripted_prompt
        elif args.prompt == "thought":
            prompt = thought_prompt
        else:
            raise NotImplementedError(f"{args.prompt} not yet implemented")
        lm = language_model.OpenAIModel(model=args.model, api_key=api_key, prompt_template="",
                                                temperature=args.temperature, top_p=args.top_p, best_of=args.best_of)
        question, answer, target = data[n]
        query = f'Q: {names[0]} asked "{question}" and {names[1]} responded "{answer}", which means?\nA:'
        prompt = prompt + query
        if args.verbose:
            print(f"Prompt: {prompt}")
        if args.use_synchromesh:
            grammar = r"""
                    ?actions: "Yes" | "No" 
                    %import common.WS
                    %ignore WS
                """
            ce = completion_engine.LarkCompletionEngine(grammar, "actions", True) 
            lm.prompt_template = prompt

            while True:
                    try:
                        response = predict_constrained(ce, lm, 1, True, stop_tokens=[".", "\n",  "Q:"])
                        break
                    except openai.error.RateLimitError as e:
                        print("Rate limit error, sleeping for 1 minute", e)
                        if total > 0:
                            print(f"Score: {score}/{total} = {score/total}")
                        time.sleep(60)
                        continue
        else:
            while True:
                try:
                    response = lm.predict_unconstrained(prompt, args.max_tokens, stop=["Q:","~"])
                    break
                except openai.error.RateLimitError as e:
                    print("Rate limit error, sleeping for 1 minute", e)
                    if total > 0:
                        print(f"Score: {score}/{total} = {score/total}")
                    time.sleep(60)
                    continue
            if args.verbose:
                print(f"Raw Response: {response}")
            raw_response = response
            response = response.split('\n')[-1].split(":")[1].strip().lower()

        if args.verbose:
            print(f"Response: {response}, target: {target}")

        target = target[:-1].lower()
        if len(target)>2:
            if "yes" in target[:3]:
                target = "yes"
            elif "no" in target[:3]:
                target = "no"
        response = response.strip().lower()
        if args.use_synchromesh:
            if response[0] == target:
                score += 1
        else:
            if target == response:
                score += 1
            
        total += 1
        print(f"Score: {score}/{total} = {score/total}")
        with open(f"{args.output_dir}/implicature_{args.dataset}_{total+args.offset}.txt", "w") as f:
            f.write(f"{prompt}\n{raw_response}\n{response}\n{target}\n{response == target}")
    print(f"Final Score: {score}/{total} = {score/total}")

if __name__ == "__main__":
    main()
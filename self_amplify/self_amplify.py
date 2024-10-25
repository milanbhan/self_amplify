
##to do : avancer sur les autres fonctions, faire une fonction generate custom pour simplifier le code

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import captum
from captum.attr import (
    LayerIntegratedGradients,
    LayerDeepLift,
    LayerGradientXActivation,
    LayerFeatureAblation,
    ShapleyValues,
    ShapleyValueSampling,
    FeatureAblation,
    Lime,
    KernelShap,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    #                       TextTemplateFeature,
    ProductBaselines,
)


class CustomWrapper(nn.Module):
    def __init__(self, model):
        super(CustomWrapper, self).__init__()
        self.transformer = model

    def forward(self, x):
        return self.transformer(x).logits[:, -1, :]


class self_amplifier:
    def __init__(self, model, tokenizer, device, stop_words=None):

        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.device = device

        if self.stop_words == None:
            self.stop_words = [
                "-", ".", ",", ";", "!", "?", "'", ":", "’", ";,", "___", "_", "(A)", "(B)", "(C)", "(D)", "(E)", "(F)",
                "(a)", "(b)", "(c)", "(d)", "(e)", "(f)" "the", "a", "to", "is", "of", "on", "in", "are", "and", "does",
            ]

        if "gemma" in tokenizer.name_or_path:
            self.user_token = "<start_of_turn>user"
            self.assistant_token = "<start_of_turn>model"
            self.end_of_turn = "<end_of_turn>"
            self.stop_token = "<eos>"
            self.correct_cst = 2
            self.layer = model.model.embed_tokens
        elif "mistral" in tokenizer.name_or_path:
            self.user_token = "[INST]"
            self.assistant_token = "[/INST]"
            self.end_of_turn = "</s>"
            self.stop_token = "</s>"
            self.correct_cst = 1
            self.layer = model.model.embed_tokens
        else:
            raise Exception("Sorry, this tokenizer is not handled so far")

    def preprocess(self, text, with_bracket):

        messages = [{"role": "user", "content": text}]

        if with_bracket:
            messages.append({"role": "assistant", "content": "The answer is ("})
            encoded_input = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            )
            # remove <\s>
            encoded_input = torch.reshape(
                encoded_input[0][: -self.correct_cst],
                (1, encoded_input[0][: -self.correct_cst].shape[0]),
            )
        else:
            encoded_input = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            )

        prompt_template = self.tokenizer.batch_decode(encoded_input)[0]

        return (prompt_template, encoded_input)

    def preprocess_self_topk(self, text, key, topk):

        important_words = ""
        for i in range(topk):
            if i == topk - 1:
                important_words += f'and "word{i+1}"'
            else:
                important_words += f'"word{i+1}", '

        preprompt = f"""Choose the right answer with the {topk} most important keywords used to answer
    Example: The answer is (A), the {topk} most important keywords to make the prediction are {important_words}.{self.stop_token}"""
        messages = [
            {"role": "user", "content": preprompt + "\n" + text},
            {
                "role": "assistant",
                "content": f'''The answer is ({key}), the {topk} most important keywords to make the prediction are "''',
            },
        ]

        encoded_input = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        )
        # remove <\s>
        encoded_input = torch.reshape(
            encoded_input[0][: -self.correct_cst],
            (1, encoded_input[0][: -self.correct_cst].shape[0]),
        )
        prompt_template = self.tokenizer.batch_decode(encoded_input)[0]

        return (prompt_template, encoded_input)

    def preprocess_self_exp(self, text, key, n_steps=3):

        explanation_template = ""
        for i in range(n_steps):
            if i == n_steps - 1:
                explanation_template += f"and step{i+1}"
            else:
                explanation_template += f"step{i+1}, "

        preprompt = f"""Choose the right answer and generate a concise {n_steps}-step explanation, with only one sentence per step
    Example: The answer is (A), {n_steps}-step explanation: {explanation_template}{self.stop_token}"""
        messages = [
            {"role": "user", "content": preprompt + "\n" + text},
            {
                "role": "assistant",
                "content": f"""The answer is ({key}), {n_steps}-step explanation: """,
            },
        ]

        encoded_input = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        )
        # remove <\s>
        encoded_input = torch.reshape(
            encoded_input[0][: -self.correct_cst],
            (1, encoded_input[0][: -self.correct_cst].shape[0]),
        )
        prompt_template = self.tokenizer.batch_decode(encoded_input)[0]

        return (prompt_template, encoded_input)

    def preprocess_auto_cot(self, text):

        preprompt = f"""Choose the right answer by thinking step by step"""
        messages = [
            {"role": "user", "content": preprompt + "\n" + text},
            {"role": "assistant", "content": f"""Let's think step by step. """},
        ]

        encoded_input = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        )
        # remove <\s>
        encoded_input = torch.reshape(
            encoded_input[0][: -self.correct_cst],
            (1, encoded_input[0][: -self.correct_cst].shape[0]),
        )
        prompt_template = self.tokenizer.batch_decode(encoded_input)[0]

        return (prompt_template, encoded_input)


    def generate_with_gradient_explanation(
        self,
        model,
        idx,
        max_new_tokens,
        explainer,
        index_min,
        index_max,
        split_dict=None,
        temperature=1.0,
        top_k=None,
        target=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        def generate_logits_prediction(input):
            output = model(input.to(self.device)).logits[:, -1, :]
            return output

        if explainer == "LayerGradientXActivation":
            lfi = LayerGradientXActivation(generate_logits_prediction, self.layer)
        elif explainer == "LayerIntegratedGradients":
            lfi = LayerIntegratedGradients(generate_logits_prediction, self.layer)
        elif explainer == "LayerDeepLift":
            lfi = LayerDeepLift(CustomWrapper(model), self.layer)

        # initialize attribution output
        if target != None:
            print(target)
            max_new_tokens = len(self.tokenizer.encode(target)[1:])
            target_next_list = self.tokenizer.encode(target)[1:]

        attribution_list = []
        for i in range(max_new_tokens):
            if target != None:
                idx_next = torch.tensor(target_next_list[i])
            else:
                # forward the model to get the logits for the index in the sequence
                output = model(idx.to(self.device))
                # pluck the logits at the final step and scale by desired temperature
                logits = output.logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution
                #             idx_next = torch.multinomial(probs, num_samples=1)
                idx_next = torch.argmax(probs)
            # Set baseline for attribution computing
            baseline = torch.clone(idx, memory_format=torch.preserve_format).to(self.device)
            baseline[0, index_min:index_max] = self.tokenizer.encode(
                self.tokenizer.pad_token
            )[1]

            # Explain the generated token
            if explainer == "LayerGradientXActivation":
                attribution = lfi.attribute(idx.to(self.device), target=idx_next)
            elif explainer == "LayerIntegratedGradients":
                attribution = lfi.attribute(
                    idx.to(self.device),
                    target=idx_next,
                    return_convergence_delta=False,
                    n_steps=20,
                    baselines=baseline,
                )
            elif explainer == "LayerDeepLift":
                attribution = lfi.attribute(
                    idx.to(self.device),
                    target=idx_next,
                    return_convergence_delta=False,
                    baselines=baseline,
                )

            attribution = attribution.sum(axis=2)[0, index_min : index_max + 1]
            attribution_list.append(attribution.cpu())
            # stop if eos token encountred
            if idx_next == self.tokenizer.encode(self.tokenizer.eos_token)[1:]:
                break
            else:
                pass

        final_attributions = torch.mean(torch.stack(attribution_list), axis=0)
        # final_attributions = torch.tensor(final_attributions / final_attributions.sum())
        final_attributions = (final_attributions / final_attributions.sum())

        tokenized_prompt_clean, lfi_list = self.agregate_token_attribution(final_attributions, idx, index_min, index_max)

        return idx, [
            dict(zip(tokenized_prompt_clean, lfi_list)),
            tokenized_prompt_clean,
            lfi_list,
        ]



    def get_index_min_max(self, idx, split_dict=None) :
        if split_dict == None :
            # end_seq = self.tokenizer.encode(f' {self.assistant_token}', add_special_tokens  = False)[1:]
            end_seq = self.tokenizer.encode(f' {self.end_of_turn}', add_special_tokens  = False)[1:]
            id_end = 0
        else:
            end_seq_str = split_dict['text']
            end_seq = self.tokenizer.encode(f' {end_seq_str} ', add_special_tokens  = False)[1:]
            id_end = split_dict['idx']

        # start_seq = tokenizer.encode(f' {self.user_token}', add_special_tokens  = False)[1:]
        start_seq = self.tokenizer.encode(f' {self.user_token}\n', add_special_tokens  = False)[1:]
        #Index where LFI can be retrieved
        index_min = [i+len(start_seq) for i in range(idx.shape[1]) if idx[0,i:i+len(start_seq)].tolist() == start_seq][0]
        #mask tokens where we don't want to compute feature importance
        index_max = [i-1 for i in range(idx.shape[1]) if idx[0,i:i+len(end_seq)].tolist() == end_seq][id_end]

        return index_min, index_max
    
    def agregate_token_attribution(self, attribution, idx, index_min, index_max):
        
        tokenized_prompt_clean = self.tokenizer.decode(
            idx[0, index_min : index_max + 1]
        ).split(" ")

        tokenized_prompt = [
            self.tokenizer.decode(idx[0, c]).replace(" ", "")
            for c in range(index_min, index_max + 1)
        ]

        raw_prompt_lfi = list(attribution.detach().numpy())
        # Agregating from subtokens to tokens
        k = 0
        query = ""
        lfi_iter = 0
        lfi_list = []
        for i, j in enumerate(tokenized_prompt):
            target_token = tokenized_prompt_clean[k]
            query += j
            lfi_iter += raw_prompt_lfi[i]
            if query == target_token:
                if query.lower() in self.stop_words:
                    lfi_iter = 0
                else:
                    pass
                lfi_list.append(lfi_iter)
                k += 1
                query = ""
                lfi_iter = 0
            else:
                pass
        
        return(tokenized_prompt_clean, lfi_list)
    
    def generate_with_pertubation_explanation(self, model, idx, explainer, index_min, index_max, split_dict=None, 
                                            temperature=1.0, top_k=None, **kwargs):
    
        if explainer == 'FeatureAblation':
            lfi = FeatureAblation(model)
        elif explainer == 'Lime':
            lfi = Lime(model)
        elif explainer == 'KernelShap':
            lfi = KernelShap(model)
        elif explainer == 'ShapleyValueSampling':
            lfi = ShapleyValueSampling(model)
        elif explainer == 'ShapleyValues':
            lfi = ShapleyValues(model)
        llm_attr = LLMAttribution(lfi, self.tokenizer)
        
        text = self.tokenizer.decode(idx[0][index_min:index_max+1])
        template_str = ""
        for _ in range(len(text.split())):
            template_str+="{} "
        template_str=template_str.strip()
        prompt_template = self.tokenizer.decode(idx[0])
        template = prompt_template.replace(text, template_str)
        
        inp = TextTemplateInput(template=template,values=text.split())
        
        #Compute attribution
        attr_res = llm_attr.attribute(inp, **kwargs)
    #     generated_text = 
        new_idx = (self.tokenizer.encode(' '.join(attr_res.output_tokens).replace("▁",""), return_tensors ='pt')[0,1:]).to(self.device)
        idx = torch.cat((idx.to(self.device), torch.reshape(new_idx, (1, new_idx.shape[0]))), dim=1)
        
        attr_input_tokens = attr_res.input_tokens
        attr_seq_attr = attr_res.seq_attr.tolist()

        for i,tok in enumerate(attr_input_tokens):
            if tok.lower() in self.stop_words:
                attr_seq_attr[i]=0

        return(idx, [dict(zip(attr_input_tokens, attr_seq_attr)), attr_input_tokens, attr_seq_attr])
    
    def generate_with_explanation(self, model, idx, explainer,
                                          topk_words, target, split_dict=None,
                                          max_new_tokens=1, temperature=1.0, top_k=None, **kwargs):
    
        index_min, index_max = self.get_index_min_max(idx=idx, split_dict=split_dict)
        
        if explainer in ["FeatureAblation", "Lime", "KernelShap", "ShapleyValueSampling", "ShapleyValues"]:
            attribution = self.generate_with_pertubation_explanation(model=model, 
                                                                    idx=idx, index_min=index_min, index_max=index_max, 
                                                                    explainer=explainer, split_dict=split_dict,
                                                                    temperature=1.0, top_k=None, target=target, n_samples=350)[1]

        elif explainer in ["LayerGradientXActivation", "LayerIntegratedGradients", "LayerDeepLift"]:
            attribution = self.generate_with_gradient_explanation(model=model, idx=idx, index_min=index_min, index_max=index_max, 
                                                                            max_new_tokens=max_new_tokens, explainer=explainer, split_dict=split_dict, 
                                                                            temperature=1.0, top_k=None, target=target)[1]
        elif explainer == 'random':
            prompt_template = self.tokenizer.decode(idx[0])
            prompt_template_list = prompt_template.split(" ")
            topk_words_list = random.sample(prompt_template_list, topk_words)
        
        else:
            raise Exception("inappropriate explainer")
        
        if explainer != "random":
    #         print(attribution)
            topk_words_list = get_topk_words(attribution, topk=topk_words)
        
        return(topk_words_list)
    
    def generate_context_idx(self, model, df, nb_shot, selection_strategy):
    
        shot_list =  []
        examples_found=0
        answer_keys = np.sort(df['AnswerKey'].unique()).tolist()
        
        while examples_found<nb_shot:
            i = df[df.index.isin(shot_list)==False].sample(n=1, replace=True).index[0]
            prompt = df['question'][i]
            target = df['AnswerKey'][i]

            idx = self.preprocess(prompt, with_bracket=True)[1]
            len_input = idx.shape[1]
            outputs = model.generate(idx.to(self.device), max_new_tokens = 1, do_sample=True, num_beams=2, 
                                    no_repeat_ngram_size=2, early_stopping=True)
            answer = self.tokenizer.decode(outputs[0][len_input:])  
            if selection_strategy == 'error':
                if (target != answer.upper()) & (target.upper() in answer_keys):
                    shot_list.append(i)
                    examples_found+=1
                else:
                    pass
                
            elif selection_strategy == 'success':
                if (target == answer.upper()) & (target.upper() in answer_keys):
                    shot_list.append(i)
                    examples_found+=1
                    
            elif selection_strategy=='random':
                shot_list = df.sample(n=nb_shot, replace=True).index.tolist()
                examples_found = nb_shot
            else:
                break
                
        return(shot_list)
    
    def generate_in_context_preprompt(self, model, df, explainer, topk_words, 
                                  idx_list_fs, n_steps=3, split_dict=None,
                                  target_map_dico=None,  **kwargs):
            
        answer_preprompt = ''
        answer_keys = np.sort(df['AnswerKey'].unique()).tolist()
        for i,j in enumerate(answer_keys):
            answer_preprompt+=f' ({j}) answer{i+1}'
        
        if explainer == "self_exp":
            rationale_example = ""
            for i in range(n_steps):
                rationale_example+=f' Step{i+1}.'
            
            in_context_preprompt = f'''You are presented with multiple choice question, where choices will look like{answer_preprompt}
    generate a concise {n_steps}-step rationale providing hints and generate the right single answer
    Ouput example: {n_steps}-step rationale:{rationale_example}, therefore the answer is (A){self.stop_token}'''
        
        elif explainer == 'auto_cot':
            in_context_preprompt = f'''You are presented with multiple choice question. 
    Choose the right answer by thinking step by step'''
        
        else:
            rationale_example = ""
            for i in range(topk_words):
                if i == topk_words-1:
                    rationale_example += 'and ' + f"'word{i+1}'"
                else:
                    rationale_example += f"'word{i+1}'" + ', '
            
            in_context_preprompt = f'''You are presented with multiple choice question, where choices will look like{answer_preprompt}
    generate {topk_words} keywords providing hints and generate the right single answer
    Ouput example: The {topk_words} keywords {rationale_example} are important to predict that the answer is (A){self.stop_token}'''
    
        messages = [{"role": "user", "content": in_context_preprompt}]
        context_size = 0
    #for every index in the context 
        for i in idx_list_fs:
            prompt = df['question'][i]
            target = df['AnswerKey'][i]
            
            if context_size==0:
                messages[0]["content"]+='\n'+prompt
            else:
                messages.append({"role": "user", "content":prompt})

            if explainer=="self_topk":
                idx = self.preprocess_self_topk(prompt, target, topk=topk_words)[1]
                len_input = idx.shape[1]
                outputs = model.generate(idx.to(self.device), max_new_tokens = 300, 
                                        do_sample=True, num_beams=2, no_repeat_ngram_size=2, 
                                        early_stopping=True).tolist()
                answer = self.tokenizer.decode(outputs[0][len_input:], skip_special_tokens=True)
                answer_with_exp = f'{topk_words} important keywords to make the prediction are "{answer}. Therefore the answer is ({target}){self.stop_token}'
                messages.append({"role": "assistant", "content": answer_with_exp})                    
                context_size+=1  
                
            elif explainer=="self_exp":
                idx = self.preprocess_self_exp(prompt, target, n_steps=n_steps)[1]
                len_input = idx.shape[1]
                outputs = model.generate(idx.to(self.device), max_new_tokens = 300, do_sample=True, 
                                        num_beams=2, no_repeat_ngram_size=2, early_stopping=True)
                answer = self.tokenizer.decode(outputs[0][len_input:], skip_special_tokens=True)
                answer_with_exp = f"{n_steps}-step rationale: {answer}, therefore the answer is ({target}){self.stop_token}"
                messages.append({"role": "assistant", "content": answer_with_exp})
                context_size+=1
                
            elif explainer=='auto_cot':
                idx = self.preprocess_auto_cot(prompt)[1]
                len_input = idx.shape[1]
                outputs = model.generate(idx.to(self.device), max_new_tokens = 300, do_sample=True, 
                                        num_beams=2, no_repeat_ngram_size=2, early_stopping=True)
                answer = self.tokenizer.decode(outputs[0][len_input:], skip_special_tokens=True)
                answer_with_exp = f"Let's think step by step: {answer}{self.stop_token}"
                messages.append({"role": "assistant", "content": answer_with_exp})
                context_size+=1
                
            else:
                idx = self.preprocess(prompt, with_bracket=True)[1]
                topk_words_list = self.generate_with_explanation(model, explainer=explainer, 
                                                                idx=idx.to(self.device), max_new_tokens=1, temperature=1.0,
                                                                split_dict=split_dict, 
                                                                topk_words=topk_words,
                                                                target=target, **kwargs)
                #Formating important key words
                topk_words_str = ""
                for word in topk_words_list:
                    word = word.replace(".", "")
                    if word == topk_words_list[-1]:
                        topk_words_str += 'and ' + f"'{word}'"
                    else:
                        topk_words_str += f"'{word}'" + ', '
                        
                answer_with_exp = f"The {topk_words} keywords {topk_words_str} suggest that the answer is ({target}){self.stop_token}"
                messages.append({"role": "assistant", "content": answer_with_exp})
                context_size+=1        
            
        return(messages)
    
    def evaluate_fs_with_exp(self, df_train, df_test, model, max_new_tokens, explainer, idx_list_fs,
                          topk_words, split_dict,
                          n_steps=3, target_map_dico=None, tokenizer_proxy=None, model_proxy=None,**kwargs):
        
        result_list = []
        answer_keys = np.sort(df_train['AnswerKey'].unique()).tolist()
        
        in_context_preprompt = self.generate_in_context_preprompt(model=model, df=df_train, 
                                explainer=explainer, topk_words=topk_words, 
                                split_dict=split_dict, idx_list_fs=idx_list_fs,
                                n_steps=n_steps,
                                target_map_dico=target_map_dico,  **kwargs)

        
        for idx in df_test.index.tolist():

            prompt = df_test['question'][idx]

            if explainer == 'self_exp':
                max_new_tokens = 300
                in_context_preprompt_iter = in_context_preprompt + [{"role": "user", "content": prompt}]
                in_context_preprompt_iter.append({"role": "assistant", "content": f'{n_steps}-step rationale: '})
                
            elif explainer == 'auto_cot':
                max_new_tokens = 300
                in_context_preprompt_iter = in_context_preprompt + [{"role": "user", "content": prompt}]
                in_context_preprompt_iter.append({"role": "assistant", "content": "Let's think step by step. "})

            else:
                in_context_preprompt_iter = in_context_preprompt + [{"role": "user", "content": prompt}]
                in_context_preprompt_iter.append({"role": "assistant", "content": f'The {topk_words} keywords "'})

            key = df_test['AnswerKey'][idx]
            print(idx)
            print(prompt)
            print("------------------------------")
            print("expected anwer: " + str(key))
            encoded_input = self.tokenizer.apply_chat_template(in_context_preprompt_iter, return_tensors='pt')
            #remove <\s>
            encoded_input = torch.reshape(encoded_input[0][:-2], (1, encoded_input[0][:-2].shape[0]))
            prompt_template = self.tokenizer.batch_decode(encoded_input)[0]
            len_input = encoded_input.shape[1]
            outputs = model.generate(encoded_input.to(self.device), max_new_tokens = max_new_tokens, do_sample=True, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)
            answer = '"' + self.tokenizer.decode(outputs[0][len_input:], skip_special_tokens=True)
            print(f"LLM answer {explainer}: ")
            print("-------->", answer)
            print("-------------------------------")

            result_list.append({'id':idx, 'prompt':prompt, 'answer':answer, 'label':key})
            
        return(pd.DataFrame(result_list))
    
def get_topk_words(attribution, topk, window=(0,0)):
        topk_indx = np.sort(np.argpartition(attribution[2], -topk)[-topk:])
        top_attribution = [attribution[1][i].replace('"\"',"") for i in topk_indx.tolist()]
        return(top_attribution)
    
    
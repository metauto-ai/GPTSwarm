#!/usr/bin/env python
# -*- coding: utf-8 -*-
# modified based on https://github.com/princeton-nlp/tree-of-thought-llm/blob/ab400345c5ea39d28ea6d7d3be0e417b11113c87/src/tot/tasks/crosswords.py
import pdb
import os
import json
import re
import random
import asyncio

class MiniCrosswordsEnv:
    def __init__(self, file):
        self.file = file
        self.n = len(self.file)
        self.cache = {}
        self.idx = -1
        self.times = 0
        self.extendable = True

    def __len__(self):
        return self.n
    
    def reset(self, idx=None, board=None, status=None, steps=None):
        if idx is None:
            idx = self.idx
        else:
            self.hints = []
        self.extendable = True
        self.idx = idx
        self.data, self.board_gt = self.file[idx]
        self.board = ['_'] * 25
        self.ans = ['_____'] * 10
        self.ans_gt = self.get_ans(self.board_gt)
        self.steps = 0
        self.status = [0] * 10  # 0: unfilled; 1: filled; 2: filled then changed
        if board is not None:
            self.board = board
            self.ans = self.get_ans(self.board)
        if status is not None:
            self.status = status
        if steps is not None:
            self.steps = steps
        return self.render()
    
    async def evaluate(self, querier, get_if_correct_prompt, get_value_prompt):
        impossible_words = []
        correct_words = []
        incorrect_words = []
        tasks = []
        prompts = []

        for i, (ans, data, status) in enumerate(zip(self.ans, self.data, self.status)):
            if ans.count('_') > 0:
                if ans.count('_') < 4:
                    ans = ' '.join(ans.lower())
                    line = f'{data}: {ans}'
                    prompt = get_value_prompt(line)
                    tasks.append(querier(prompt))
                continue
            prompt = get_if_correct_prompt(ans, data)
            tasks.append(querier(prompt))
            prompts.append(prompt)
        
        await asyncio.gather(*tasks)
        for i, (ans, data, status) in enumerate(zip(self.ans, self.data, self.status)):
            idx = 'v' if i >= 5 else 'h'
            idx += str(i+1) if i < 5 else str(i-5+1)
            idx += '. '
            if ans.count('_') > 0: 
                if ans.count('_') < 4:
                    ans = ' '.join(ans.lower())
                    line = f'{data}: {ans}'
                    prompt = get_value_prompt(line)
                    res = await querier(prompt)
                    res = res.split('\n')[-1].strip()
                    if 'impossible' in res:
                        impossible_words.append((idx, ans, data))
                continue
            prompt = get_if_correct_prompt(ans, data)
            res = await querier(prompt)
            if res == 'Yes':
                correct_words.append((idx, ans, data))
            elif res == 'No':
                incorrect_words.append((idx, ans, data))
        self.correct_words = correct_words 
        self.incorrect_words = incorrect_words
        self.impossible_words = impossible_words
        return len(correct_words) / 10

    async def check_termination(self, querier, get_value_prompt):
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        self.impossible_words = []
        tasks = []
        prompts = []
        
        for i, (ans, data, status) in enumerate(zip(self.ans, self.data, self.status)):
            # if status != 0: continue
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = get_value_prompt(line)
            tasks.append(querier(prompt))
            prompts.append(prompt)

        await asyncio.gather(*tasks)

        for i, (ans, data, status) in enumerate(zip(self.ans, self.data, self.status)):
            # if status != 0: continue
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = get_value_prompt(line)
            res = await querier(prompt)
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1
            # if res == 'impossible': 
            #     if i < 5:
            #         self.impossible_words.append('h' + str(i+1) + '. ' +  self.ans[i] + ' -- ' + self.data[i])
            #     else:
            #         self.impossible_words.append('v' + str(i-5+1) + '. ' + self.ans[i] + ' -- ' + self.data[i])
        self.extendable = count['impossible'] == 0
    
    def render_gt_board(self):
        s = "GT Board:\n"
        for i in range(5):
            s += ' '.join(self.board_gt[i*5:(i+1)*5]) + '\n'
        return s
    
    def render_board(self):
        s = "Current Board:\n"
        for i in range(5):
            s += ''.join(self.board[i*5:(i+1)*5]) + '\n'
        return s

    def render_clues(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + '\n'
        return s
    
    def render_ans(self, status=None):
        s = []
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s.append('h' + str(i+1) + '. '  + self.ans[i] + ' -- ' + self.data[i] + '\n')
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s.append('v' + str(i-5+1) + '. ' + self.ans[i] + ' -- ' + self.data[i] + '\n')
        return '\n'.join(random.sample(s, len(s)))
    
    def render_gt_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.ans_gt[i] + ' -- ' + self.data[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.ans_gt[i] + ' -- ' + self.data[i] + '\n'
        return s

    def render(self, status=True, include_hints=True):
        if status:
            s = self.render_board() + '\nUnfilled:\n' + self.render_ans(status=0) + '\nFilled:\n' + self.render_ans(status=1) + '\nChanged:\n' + self.render_ans(status=2)
        else:
            s = self.render_board() + '\n' + self.render_ans()
        if include_hints and len(self.hints) > 0:
            s += '\nSuggestions:\n' + '\n'.join(self.hints[-1:])
        return s
    
    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans
    
    @property
    def r_word(self):
        return sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
    @property
    def r_letter(self):
        return sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
    @property
    def r_game(self):
        return self.board == self.board_gt

    def step(self, action, allow_change=True):
        self.steps += 1
        action = action.split('\n')[-1]
        action = action.split('. ')
        
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        pos, word = action
        word = word[:5].upper()

        if pos.startswith('h'):
            idx = int(pos[1:]) - 1
            for i in range(5):
                if (not allow_change) and self.board[idx*5+i] != '_' and self.board[idx*5+i] != word[i]:
                    raise Exception('Invalid! You cannot change a filled letter.')
            self.board[idx*5:(idx+1)*5] = list(word)
        elif pos.startswith('v'):
            idx = int(pos[1:]) - 1
            for i in range(5):
                if (not allow_change) and self.board[idx+i*5] != '_' and self.board[idx+i*5] != word[i]:
                    raise Exception('Invalid! You cannot change a filled letter.')
            self.board[idx::5] = list(word)
            idx += 5  # for later status update
        else:
            return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}
        
        self.new_ans = self.get_ans(self.board)
        # self.status = [2 if (status == 1 and ans != new_ans) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status = [2 if any(letter != new_letter and letter != '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status[idx] = 1
        self.ans = self.new_ans
        
        return self.render(), self.r_game, (self.r_game or self.steps >= 20), {'r_letter': self.r_letter, 'r_word': self.r_word, 'r_game': self.r_game}
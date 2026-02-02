#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from googleapiclient.discovery import build
import requests
import ast


class BoChaSearchEngine():
    def __init__(self) -> None:
        self.api_key = os.getenv("BOCHA_API_KEY")

    def search(self, query: str, num: int = 3):
        try:
            url = 'https://api.bochaai.com/v1/web-search'
            headers = {
                'Authorization': f'Bearer {self.api_key}',  # 请替换为你的API密钥
                'Content-Type': 'application/json'
            }
            data = {
                "query": query,
            }

            response = requests.post(url, headers=headers, json=data)

            json_response = response.json()
            res = json_response["data"]["webPages"]["value"]
            print(res)
            return '\n'.join([item['snippet'] for item in res])
        except:
            return 'Cannot get search results from BoCha API'

class BingSearchEngine():
    def __init__(self) -> None:
        self.api_key = os.getenv("BING_API_KEY")
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.headers = {"Ocp-Apim-Subscription-Key": self.api_key}
    
    def search(self, query: str, num: int = 3):
        try:
            params = {"q": query, "count": num}
            res = requests.get(self.endpoint, headers=self.headers, params=params)
            res = res.json()
            return '\n'.join([item['snippet'] for item in res['webPages']['value']])
        except:
            return 'Cannot get search results from Bing API'

class GoogleSearchEngine():
    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cse_id = os.getenv("GOOGLE_CSE_ID")
        self.service = build("customsearch", "v1", developerKey=self.api_key)
        
    def search(self, query: str, num: int = 3):
        try:
            res = self.service.cse().list(q=query, cx=self.cse_id, num=num).execute()
            return '\n'.join([item['snippet'] for item in res['items']])
        except:
            return 'Cannot get search results from Google API'

class SearchAPIEngine():

    def search(self, query: str, item_num: int = 3):
            try:
                url = "https://www.searchapi.io/api/v1/search"
                params = {
                "engine": "google",
                "q": query,
                "api_key": os.getenv("SEARCHAPI_API_KEY")
                }

                response = ast.literal_eval(requests.get(url, params = params).text)

            except:
                return ''
            
            if 'knowledge_graph' in response.keys() and 'description' in response['knowledge_graph'].keys():
                return response['knowledge_graph']['description']
            if 'organic_results' in response.keys() and len(response['organic_results']) > 0:
                
                return '\n'.join([res['snippet'] for res in response['organic_results'][:item_num]])
            return 'Cannot get search results from SearchAPI'



if __name__ == "__main__":
    # search_engine = GoogleSearchEngine()
    # print(search_engine.search("cell phone tower"))

    print(SearchAPIEngine().search("Juergen Schmidhuber"))
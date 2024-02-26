
from swarm.graph import Node
from swarm.llm.format import Message


class CrosswordsOperation(Node):
    async def llm_query_with_cache(self, prompt):
        cache = self.memory.query_by_id("cache")
        if len(cache) == 0:
            cache = {}
            self.memory.add("cache", cache)
        else:
            cache = cache[0]
        if not prompt in cache.keys():    
            cache[prompt] = await self.llm.agen([Message(role="user", content=prompt)], temperature=0.0)
        return cache[prompt]
import json

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage

from agentchat.core.models.manager import ModelManager
from agentchat.prompts.rewrite import system_query_rewrite
from agentchat.prompts.rewrite import user_query_write

class QueryRewrite:
    def __init__(self):
        self.client = ModelManager.get_conversation_model()

    async def rewrite(self, user_input):
        rewrite_prompt = user_query_write.format(user_input=user_input)
        response = await self.client.ainvoke([
            SystemMessage(content=system_query_rewrite),
            HumanMessage(content=rewrite_prompt),
        ])
        cleaned_response = response.content.replace("```json", "")
        cleaned_response = cleaned_response.replace("```", "").strip()

        try:
            result = json.loads(cleaned_response)
            if isinstance(result, dict):
                return result.get("variations", [])
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.info(f"json loads error: {e}")
            return [user_input]

query_rewriter = QueryRewrite()

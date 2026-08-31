import requests
from loguru import logger
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
from uuid import uuid4

from langchain.tools import tool

from agentchat.settings import app_settings
from agentchat.services.storage import storage_client

@tool(parse_docstring=True)
def text_to_image(user_prompt: str):
    """
    根据用户提供的提示词产生图片。

    Args:
        user_prompt (str): 用户的图片提示词。

    Returns:
        str: 生成的图片链接。
    """
    return _text_to_image(user_prompt)


def _text_to_image(user_prompt):
    """给用户的图片描述生成一张照片"""
    model_config = app_settings.multi_models.text2image
    payload = {
        **model_config.parameters,
        "model": model_config.model_name,
        "prompt": user_prompt,
    }
    headers = {
        "Authorization": f"Bearer {model_config.api_key}",
        "Content-Type": "application/json",
    }

    try:
        generation_response = requests.post(
            model_config.base_url,
            headers=headers,
            json=payload,
            timeout=300,
        )
        generation_response.raise_for_status()
        images = generation_response.json().get("images", [])
        if not images or not images[0].get("url"):
            raise ValueError("SiliconFlow 生图响应中没有图片 URL")

        image_url = images[0]["url"]
        url_path = urlparse(image_url).path
        unquoted_path = unquote(url_path)
        file_name = PurePosixPath(unquoted_path).name or f"{uuid4().hex}.png"
        oss_object_name = f"text_to_image/{file_name}"

        image_response = requests.get(image_url, timeout=60)
        image_response.raise_for_status()
        storage_client.upload_file(oss_object_name, image_response.content)
        logger.info(f"图片 {file_name} 已成功上传到对象存储")

        storage_base_url = app_settings.storage.active.base_url.rstrip("/")
        return f"您的图片已经生成完毕，图片链接为：![图片]({storage_base_url}/{oss_object_name})"
    except Exception as error:
        logger.error(f"SiliconFlow 图片生成或存储失败: {error}")
        raise ValueError(f"SiliconFlow 图片生成或存储失败: {error}") from error

import requests
from loguru import logger
from langchain.tools import tool

from agentchat.settings import app_settings
from agentchat.prompts.tool import WEATHER_PROMPT, MESSAGE_PROMPT


@tool(parse_docstring=True)
def get_weather(city: str):
    """
    查询用户提供城市的天气情况。

    Args:
        city (str): 用户提供的城市名称。

    Returns:
        str: 城市的天气信息。
    """
    return _get_weather(city)

def _get_weather(location: str):
    """帮助用户想要查询的天气"""
    params = {
        'key': app_settings.tools.weather.get('api_key'),
        'city': location,
        'extensions': 'all'
    }

    try:
        res = requests.get(url=app_settings.tools.weather.get('endpoint'), params=params, timeout=5)  # 预报天气
        res.raise_for_status()
        result = res.json()

        if not isinstance(result, dict):
            raise ValueError("天气服务返回了无法识别的数据格式")

        if str(result.get("status")) != "1":
            info = result.get("info") or "未知错误"
            infocode = result.get("infocode") or "unknown"
            raise ValueError(f"天气服务请求失败：{info}（错误码：{infocode}）")

        forecasts = result.get("forecasts") or []
        if not forecasts or not isinstance(forecasts[0], dict):
            raise ValueError(f"天气服务没有返回“{location}”的预报数据，请尝试使用城市名或行政区编码")

        forecast = forecasts[0]
        city = forecast.get("city") or location  # 获取城市
        message_result = []
        data = forecast.get("casts") or []

        if not data:
            raise ValueError(f"天气服务没有返回“{city}”的逐日预报数据")

        for item in data:
            date = item.get('date')  # 获取日期
            day_temp = item.get('daytemp')  # 白天温度
            night_temp = item.get('nighttemp')  # 晚上温度
            day_weather = item.get('dayweather')  # 白天天气现象
            night_weather = item.get('nightweather')  # 晚上天气现象
            weather_message = MESSAGE_PROMPT.format(date, day_temp, night_temp, day_weather, night_weather)

            message_result.append(weather_message)

        final_result = WEATHER_PROMPT.format(city, message_result[0], message_result[1:])
        return final_result
    except Exception as err:
        error_message = f"查询“{location}”天气失败：{err}"
        logger.error(f'Call Weather Tool Err: {error_message}')
        return error_message



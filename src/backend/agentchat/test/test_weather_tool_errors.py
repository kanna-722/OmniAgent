from agentchat.tools.get_weather.action import _get_weather


class StubResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_weather_reports_provider_error_instead_of_none_type(monkeypatch):
    monkeypatch.setattr(
        "agentchat.tools.get_weather.action.requests.get",
        lambda **kwargs: StubResponse({
            "status": "0",
            "info": "INVALID_USER_KEY",
            "infocode": "10001",
        }),
    )

    result = _get_weather("成都市成华区")

    assert "INVALID_USER_KEY" in result
    assert "10001" in result
    assert "NoneType" not in result


def test_weather_reports_empty_forecast(monkeypatch):
    monkeypatch.setattr(
        "agentchat.tools.get_weather.action.requests.get",
        lambda **kwargs: StubResponse({"status": "1", "forecasts": []}),
    )

    result = _get_weather("不存在的地区")

    assert "没有返回" in result
    assert "NoneType" not in result

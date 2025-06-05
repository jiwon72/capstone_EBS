import os
import openai

def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 256) -> str:
    """
    OpenAI API를 호출하여 프롬프트에 대한 전문가 설명을 한글로 반환합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[OpenAI API 키가 설정되어 있지 않습니다.]"
    try:
        client = openai.OpenAI(api_key=api_key)
        # 한글 답변을 유도하는 시스템 메시지 추가
        messages = [
            {"role": "system", "content": "너는 한국어로 답변하는 투자 전문가야. 모든 답변은 반드시 한글로 작성해."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI API 호출 오류: {str(e)}]" 
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

load_dotenv()


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL")
)


def main():
    response: ChatCompletion = client.chat.completions.create(
        model="qwen3:30b-a3b-nothinking",
        messages=[
            {"role": "system", "content": "你是一个友好的聊天机器人"},
            {"role": "user", "content": "请解释一下机器学习的基本概念"},
        ],
        max_tokens=1024,
        temperature=0.5,
    )

    print(f"AI回复: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()

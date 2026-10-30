from dotenv import load_dotenv
load_dotenv()

import litellm

resp = litellm.completion(
    model="gemini/gemini-3.5-flash-lite",
    messages=[{"role": "user", "content": "قل مرحبا بكلمة واحدة"}],
)
print("الرد:", resp.choices[0].message.content)
print("التوكنز:", resp.usage.total_tokens)
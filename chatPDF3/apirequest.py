import openai

openai.api_key = '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'

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": "'Translate this English text to French: Hello, my name is John.'"},
    ]
)

# print(response.choices[0].message['content'])
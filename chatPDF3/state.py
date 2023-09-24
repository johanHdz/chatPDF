# state.py
import reflex as rx
import os
import openai

class State(rx.State):
    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    openai.api_key = "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"
    def answer(self):
        # Our chatbot has some brains now!
        session = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                # {"role": "user", "content": self.question}
                {"role": "user", "content": "Translate this English text to Spain: Hi, my name is John,"}
            ],
            #stop=None,
            #temperature=0.7,
            #stream=True,
        )
        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))
        self.chat_history.append("Hola, soy un chatbot")
        self.chat_history.append("Lo siento, como IA no puedo responder eso.")

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        for item in session:
            if hasattr(item.choices[0].delta, "content"):
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (
                    self.chat_history[-1][0],
                    answer,
                )
                yield
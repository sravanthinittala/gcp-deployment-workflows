from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, request

llm = ChatGoogleGenerativeAI(model="gemini-flash-2.0")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about a given input"
        ),
        ("human","{input}")
    ]
)

chain = prompt | llm

def create_app():
    app = Flask(__name__)

    @app.route("/ask", methods=['POST'])
    def talkToGemini():
        user_input = request.json['input']
        response = chain.invoke({"input": user_input})
        return response.content

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=80)
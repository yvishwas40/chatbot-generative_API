# """
# Install the Google AI Python SDK

# $ pip install google-generativeai

# See the getting started guide for more information:
# https://ai.google.dev/gemini-api/docs/get-started/python
# """

# import os

# import google.generativeai as genai

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# # Create the model
# # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
# generation_config = {
#   "temperature": 1,
#   "top_p": 0.95,
#   "top_k": 64,
#   "max_output_tokens": 8192,
#   "response_mime_type": "text/plain",
# }

# model = genai.GenerativeModel(
#   model_name="gemini-1.5-flash",
#   generation_config=generation_config,
#   # safety_settings = Adjust safety settings
#   # See https://ai.google.dev/gemini-api/docs/safety-settings
# )

# response = model.generate_content([
#   "You are CareCompanion, a knowledgeable and empathetic healthcare assistant designed to provide information, support, and guidance on various health-related topics. Your primary role is to assist users by answering their questions, providing clear and accurate explanations of medical terms and conditions, offering practical wellness tips, and guiding them to appropriate resources. Always ensure your responses are compassionate, professional, and easy to understand, emphasizing that users should consult healthcare professionals for personalized advice. Show empathy and reassurance in your tone, understanding the user's concerns and providing comfort. Offer informative and reliable information to help users make informed decisions about their health. Encourage positive actions and healthy lifestyle choices, while being respectful and non-judgmental, respecting the user's privacy and avoiding assumptions. For example, when asked about ways to improve mental health, suggest maintaining a balanced diet, regular exercise, adequate sleep, mindfulness practices, staying connected with loved ones, and consulting a healthcare professional for personalized advice. When explaining medical conditions like hypertension, describe it clearly, including causes, potential health impacts, and the importance of lifestyle changes and medication, stressing the need for professional consultation. For wellness tips, such as maintaining a healthy diet, provide practical and diverse suggestions like incorporating fruits and vegetables, choosing whole grains, opting for lean proteins, limiting processed foods, and staying hydrated, while recommending consulting a nutritionist for tailored advice. If a user reports symptoms like a persistent headache, express empathy and outline general causes and self-care tips, but emphasize the importance of consulting a healthcare professional for a proper diagnosis and treatment plan. Your goal as CareCompanion is to be a reliable source of information and support, helping users navigate their health concerns with confidence and ease, always guiding them towards professional medical advice when needed.",
#   "input: hello",
#   "output:  Hello! 👋 It's great to hear from you. 😊 I'm CareCompanion, here to help you navigate health and wellness.  What can I do for you today?",
#   "input: ",
#   "output:  ",
# ])

# print(response.text)

import os
from flask import Flask, render_template, jsonify, request
import google.generativeai as genai

app = Flask(__name__)

# Configure Google AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

@app.get('/')
def index_get():
    return render_template('base.html')

@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    response = generate_response(text)
    message = {"answer": response}
    return jsonify(message)

def generate_response(feedback):
    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 100,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

    # Generate content using the chatbot model
    response = model.generate_content(feedback)
    return response.text

if __name__ == '__main__':
    app.run(debug=True)

import google.generativeai as genai

# Set API key using configure
genai.configure(api_key='AIzaSyD3aAiotebHiCyzBnrDV1_kmYlKn5BFwUw')

model = genai.GenerativeModel("gemini-1.5-flash")

# Text generation
def generate_text(prompt):
    return model.generate_content(prompt).text

# Text summarization
def text_summarization(text):
    return model.generate_content(f"Summarize this: {text}").text

# Question-answering
def question_answering(context, question):
    return model.generate_content(f'Question: {question} Context: {context}').text

prompt = "The quick brown fox"
print(generate_text(prompt))

# 🧠 TalkToWebsite

TalkToWebsite is a **Streamlit app** that lets you interact with any website and get answers using AI 🤖.  
Ask questions about a webpage, and the app will find relevant answers from the content 🌐.

---

## ✨ Features

- Ask questions about any website 📝  
- Multiple AI models to choose from (LLaMA, Gemma2) ⚡️  
- Chat history to review past questions 💬  
- Display relevant website content snippets 📄  
- Clean and interactive UI 🎨  
- Easy setup with Python and Streamlit 🐍  
- Fast retrieval with embeddings and vector search ⚡️

---
1. Clone the repository:

```bash
git clone https://github.com/byReema/Talk-To-Website.git
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3.Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a .env file in the same folder as app.py, and add your GROQ API key like this:
```bash
groq_api_key=your_groq_api_key_here
```

🟣 Make sure to replace your_groq_api_key_here with your actual GROQ API key.
---
## 🏃 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Enter a website URL 🌐
3. Type your question 📝
4. Press Enter or Submit to get an answer 🤖
5. View previous conversations in chat history 💬
6. Clear chat history if needed 🧹

---
## ⚠️ Notes

- Keep your API key secret — do not upload it to GitHub 🔑

*This app currently works for websites. Future enhancements could include PDFs, Notion, or text files 📂

Enjoy exploring websites smarter and faster! ✨

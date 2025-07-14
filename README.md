# AI Research Paper Assistant

This Streamlit application helps you summarize and ask questions about one or more research papers in PDF format. It uses the OpenRouter API for language model interactions and provides features like multi-document comparison, question-answering based on semantic search, and ROUGE score evaluation for summaries.

## Features

-   **Upload Multiple PDFs**: Analyze several papers in one session.
-   **AI-Powered Summaries**: Generate a concise summary for each paper.
-   **Compare Summaries**: If you upload more than one paper, the app provides a side-by-side comparison and a comparative summary.
-   **Question & Answer**: Ask specific questions about a selected paper. The AI will answer based on the most relevant sections of the document.
-   **View Context Chunks**: See the exact text chunk and page image that the AI used to answer your question.
-   **ROUGE Score Evaluation**: Automatically calculates ROUGE-1, ROUGE-2, and ROUGE-L F1-scores to evaluate the quality of the generated summary against the source text.

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

The application requires an API key from [OpenRouter](https://openrouter.ai/).

1.  Create a file named `.env` in the root of the project directory.
2.  Add your API key to the `.env` file as follows:

```
OPENROUTER_API_KEY="your-openrouter-api-key"
```

## How to Run

Once you have completed the setup, you can run the Streamlit app with the following command:

```bash
streamlit run model.py
```
The app is easy to deploy via the streamlit community login page, our version of the website is https://research-summarizer.streamlit.app/
The application will open in a new tab in your web browser. 

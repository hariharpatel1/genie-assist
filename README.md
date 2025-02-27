# GenieAssist
An intelligent assistant for onboarding new team members and answering questions about your codebase, documentation, and processes.

## 🚀 Features

- **Documentation Search**: Quickly find relevant information in Google Docs, PDFs, and other documents
- **Code Explorer**: Search and understand your codebase across multiple repositories
- **Guided Onboarding**: Step-by-step guidance for new team members
- **Human-in-the-Loop**: Escalation to human experts for complex questions
- **Conversation Memory**: Maintains context across multiple interactions
- **Modern Web UI**: Clean, responsive Streamlit interface

## 📋 Requirements

- Python 3.9+
- Azure OpenAI API access
- GitHub repository access (for code exploration)
- Google Cloud credentials (for Google Docs access, optional)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-org/onboarding-agent.git
cd onboarding-agent
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Copy the environment variables example file and edit as needed:

```bash
cp .env.example .env
```

5. Edit the `.env` file with your API keys and configuration.

## 🚀 Usage

### Running the Streamlit App

Start the Streamlit app:

```bash
streamlit run app.py
```

This will open the assistant in your web browser (typically at `http://localhost:8501`).

### Indexing Your Content

Before using the assistant, you need to index your documentation and code repositories. You can do this programmatically:

```python
from retrieval.document_retriever import DocumentRetriever
from retrieval.code_retriever import CodeRetriever

# Index Google Docs
doc_retriever = DocumentRetriever()
doc_retriever.load_and_index_google_docs()

# Index PDF documentation
doc_retriever.load_and_index_pdf_directory("path/to/pdfs")

# Index code repositories
code_retriever = CodeRetriever()
code_retriever.load_and_index_repositories()
```

## 🧩 Extending the Agent

### Adding New Tools

1. Create a new tool file in the `tools/` directory
2. Implement your tool following the LangChain Tool pattern
3. Add your tool to the list in `agents/onboarding_agent.py`

### Adding New Data Sources

To add new data sources:

1. Create a new loader in the `retrieval/loaders/` directory
2. Implement the loader following the pattern of existing loaders
3. Add the new loader to the appropriate retriever class

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Resources

- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [Streamlit Documentation](https://docs.streamlit.io/)
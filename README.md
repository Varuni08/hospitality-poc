\# Multi-Agent Hospitality AI Assistant



An AI-powered restaurant assistant that answers customer questions, searches menu data, and manages reservation workflows through a multi-agent architecture.



\## Features



\* Answers questions about restaurant information and policies

\* Searches menu items by category, price, ingredients and preferences

\* Creates, looks up, modifies and cancels reservations

\* Handles callback requests

\* Routes requests to specialised agents

\* Uses separate RAG pipelines for restaurant information and menu search

\* Generates grounded responses using retrieved context

\* Stores and manages reservation data through Supabase



\## Architecture



The application uses an orchestrator to identify the user’s intent and route the request to the appropriate agent:



\* `info\_agent.py` — restaurant information queries

\* `menu\_agent.py` — menu search and filtering

\* `reservation\_agent.py` — reservation and callback operations

\* `orchestrator.py` — intent detection and agent routing



\## Tech Stack



\* Python

\* Retrieval-Augmented Generation

\* Pinecone

\* Groq LLM

\* SentenceTransformers

\* LangChain

\* Supabase

\* Pandas

\* PDF and CSV processing

\* Streamlit



\## Project Structure



\* `agents/` — specialised information, menu and reservation agents

\* `app.py` — application interface

\* `orchestrator.py` — request routing logic

\* `info\_restaurant.py` — restaurant-information RAG pipeline

\* `menu.py` — menu retrieval and filtering

\* `reservation.py` — reservation workflow

\* `config.py` — environment-based configuration

\* `Fake restaurant info.pdf` — synthetic restaurant knowledge base

\* `saffron\_table\_menu\_dataset\_ffs\_bb.csv` — synthetic menu dataset



\## Setup



Install the dependencies:



```

pip install -r requirements.txt

```



Create a `.env` file using `.env.example` and add the required credentials:



```

GROQ\_API\_KEY=your\_groq\_api\_key

PINECONE\_API\_KEY=your\_pinecone\_api\_key

SUPABASE\_URL=your\_supabase\_url

SUPABASE\_KEY=your\_supabase\_key

```



Run the application:



```

streamlit run app.py

```



\## Security



Credentials are loaded through environment variables and are not stored in the repository.



\## Note



The restaurant information and menu data included in this repository are synthetic and intended for demonstration purposes.


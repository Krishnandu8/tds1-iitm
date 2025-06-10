# TDS Project: Virtual Teaching Assistant API

This repository contains the code for the Virtual Teaching Assistant project for IIT Madras Online Degree in Data Science, Tools in Data Science (TDS) course.

The goal is to create an API that can automatically answer student questions based on course content and Discourse forum posts using Retrieval-Augmented Generation (RAG).

---

## Project Structure

* `data/`: This directory will eventually store raw scraped data. **Crucially, it also contains the `chroma_db/` subdirectory, which holds your pre-built ChromaDB vector store.** This database, containing the vectorized course content and forum posts, is committed to the repository for deployment.
* `src/`: (Currently unused in this structure, but can contain core API and logic if split further).
* `scripts/`: Contains scripts for data processing and vector database creation, notably `process_data.py`.
* `README.md`: This file.
* `LICENSE`: MIT License.
* `requirements.txt`: Python dependencies for the project.
* `.gitignore`: Specifies files and directories that Git should ignore (e.g., virtual environments, sensitive data).
* `venv/`: Python virtual environment (ignored by `.gitignore`).
* `.env`: **(Sensitive file)** This file stores environment variables like your OpenAI API key and is crucial for the application's functionality. It is ignored by `.gitignore`.
* `auth.json`: **(Sensitive file)** If used for authentication state (e.g., Playwright or other tools), this file is also ignored by `.gitignore`.

---

## Setup

Follow these steps to set up the project locally:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/tds1-iitm.git](https://github.com/your-username/tds1-iitm.git) # Replace with your actual repo URL
    cd tds1-iitm
    ```

2.  **Create and Activate a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the root of your project directory (the same level as `main.py` and `requirements.txt`). Add your OpenAI API key to this file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    # Optional: If you use a custom base URL for OpenAI API (e.g., for AI proxy)
    # OPENAI_BASE_URL="[https://api.example.com/v1](https://api.example.com/v1)"
    ```
    **Important:** The `.env` file is listed in `.gitignore` and should **never** be committed to your public repository.

5.  **Prepare the Data (ChromaDB):**
    The application relies on a pre-built ChromaDB vector store.
    * The `scripts/process_data.py` script is designed to scrape data, process it, and populate the ChromaDB.
    * **To generate the database locally:**
        * Open `scripts/process_data.py` and ensure `CHROMA_DB_PATH` is set to a local, temporary path like `./local_chroma_db` for local execution.
        * Run the script:
            ```bash
            uv run scripts/process_data.py
            ```
        * After the script completes, **copy the contents** of the generated `local_chroma_db` directory into the `chroma_db/` directory in your project root.
        * **Crucial for deployment:** The `chroma_db/` folder (containing the database files like `chroma.sqlite3`) **must be committed to your Git repository**. This is how your deployed application on Render (or similar platforms without shell access) will find the database.

---

## Usage

Once the setup is complete and your `chroma_db/` is populated, you can run the FastAPI application:

1.  **Start the FastAPI Application:**
    Ensure your virtual environment is active.
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 10000
    ```
    This will start the server, typically accessible at `http://0.0.0.0:10000`.

2.  **Access the API:**
    * **API Documentation:** You can access the interactive API documentation (Swagger UI) by navigating to `/docs` in your browser (e.g., `http://localhost:10000/docs`).
    * **Query Endpoint:** The main endpoint for asking questions is `POST /`. You can test this using the `/docs` interface or a tool like `curl` or Postman.
        * **Example Request Body (JSON):**
            ```json
            {
              "question": "What are the deadlines for the upcoming assignments?"
            }
            ```

---

## Sensitive Data & Git Management

It is critical to keep sensitive information like API keys (`.env`) and authentication tokens (`auth.json`) out of your public Git repository.

* Your `.gitignore` file includes entries for `.env` and `auth.json` (and `chroma_db/` if you're not committing it).
* **If you accidentally committed these files before adding them to `.gitignore`:**
    You must remove them from Git's tracking:
    ```bash
    git rm --cached .env
    git rm --cached auth.json
    git commit -m "Stop tracking sensitive files"
    git push origin main --force # Use --force with caution on shared repos
    ```
    This removes them from your repository's history on GitHub but keeps them on your local machine.

---

## Deployment

This application is designed to be deployed to platforms like Render. The key considerations for deployment are:
* Ensuring `requirements.txt` is up-to-date.
* Having the `chroma_db/` directory committed to your repository so the deployed application can load the database.
* Configuring environment variables (like `OPENAI_API_KEY`) directly on the deployment platform.

---

Feel free to ask if you need further clarification on any part of this!
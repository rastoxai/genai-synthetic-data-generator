# Synthetic Data Generator (Gemini + SDV)

This project provides a Streamlit web application for generating synthetic data for multiple related tables. It leverages the Gemini API to create an initial small sample of realistic data with referential integrity and then uses the Synthetic Data Vault (SDV) library's HMA Synthesizer to generate a larger synthetic dataset based on the patterns learned from the initial sample.

---

## Features

* **Schema Input**: Define multiple table schemas easily using comma or space-separated column names via a simple Streamlit UI.
* **Initial Data Generation**: Uses the Gemini API to generate a small, realistic sample dataset (10–20 rows per table) that respects referential integrity based on column names.
* **Metadata Inference**: SDV automatically infers data types and relationships (primary/foreign keys) from the Gemini-generated sample data.
* **Synthetic Data Synthesis**: SDV's HMA Synthesizer learns the statistical properties and relationships from the sample and generates a larger synthetic dataset.
* **Scalability**: Control the size of the synthetic dataset using a scale factor relative to the initial sample size.
* **Data Download**: Download the generated synthetic data for each table as CSV files.

---

## Project Structure

```
your_project_folder/
├── .env                 # Contains your GEMINI_API_KEY (DO NOT commit this!)
├── .gitignore           # Specifies files/folders to ignore (like .env)
├── requirements.txt     # Lists project dependencies
├── app.py               # The main Streamlit application code
└── tests/               # Directory for test files
    └── test_app.py      # Pytest test cases
```

---

## Prerequisites

Before running the application, ensure you have the following installed:

* Python 3.7 or higher
* A Google Cloud or Google AI Platform account with access to the Gemini API
* A Gemini API Key

---

## Setup

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd your_project_folder
   ```

   Replace `<repository_url>` with the actual URL of your repository.

2. **Create a virtual environment (recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Gemini API Key**:

   * Create a file named `.env` in the root directory and add:

     ```env
     GEMINI_API_KEY='YOUR_ACTUAL_GEMINI_API_KEY'
     ```
   * Replace `'YOUR_ACTUAL_GEMINI_API_KEY'` with your real API key.
   * Ensure `.env` is listed in `.gitignore`.

---

## Running the Application

Ensure your virtual environment is activated:

```bash
source venv/bin/activate
```

Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## Running the Tests

Basic unit tests using `pytest` are included.

Ensure your virtual environment is activated:

```bash
source venv/bin/activate
```

Run the tests from the project root:

```bash
pytest
```

Pytest will execute the tests in the `tests/` directory and display results in the terminal.

---

## Contributing

<!-- Optional: Add details on pull request process, code style guidelines, etc. -->

---

## License

<!-- Optional: Add license information here -->


# GEMINI.md

## Project Overview

This project is a stock prediction API built with Python and FastAPI. It provides an endpoint to predict future stock prices based on historical data. The API is containerized using Docker for easy deployment.

**Key Technologies:**

*   **Backend:** FastAPI
*   **Data Validation:** Pydantic
*   **Machine Learning:** Scikit-learn, TensorFlow
*   **Dependencies:** `fastapi`, `uvicorn`, `pydantic`, `numpy`, `scikit-learn`, `joblib`, `python-dateutil`, `huggingface-hub`, `tensorflow-cpu`

**Architecture:**

The project follows a clean architecture with a clear separation of concerns:

*   `main.py`: The main application entry point.
*   `routes/`: Defines the API endpoints.
*   `services/`: Contains the business logic for prediction and validation.
*   `models/`: Defines the Pydantic models for request and response data.
*   `predict/`: Contains the core prediction logic.
*   `utils/`: Provides utility functions for data preprocessing and formatting.
*   `config/`: Contains configuration for the application.

## Building and Running

### Local Development

1.  **Prerequisites:**
    *   Python 3.11+
    *   uv

2.  **Installation:**
    ```bash
    # Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # Install dependencies
    uv pip install -r requirements.txt
    ```

3.  **Running the Application:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

### Docker

1.  **Build the Docker Image:**
    ```bash
    docker build -t amiyokm/stock-price-predictor .
    ```

2.  **Run the Docker Container:**
    ```bash
    docker run -d -p 8000:8000 amiyokm/stock-price-predictor
    ```

The API will be available at `http://localhost:8000`.

## Development Conventions

*   **Code Style:** The project follows the standard PEP 8 style guide for Python.
*   **Type Hinting:** All functions and methods have type hints for better code clarity and validation.
*   **Modularity:** The code is organized into modules with specific responsibilities, promoting reusability and maintainability.
*   **Validation:** Pydantic is used for robust data validation of API requests.
*   **Testing:** (TODO: Add information about testing practices if available)

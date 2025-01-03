Set-up Virtual environment

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

   Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   Run the backend
   
   ```bash
   uvicorn main:app
   ```
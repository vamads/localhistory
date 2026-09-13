# Local History web client

A focused React interface for the existing retrieval system. Articles with known
coordinates appear on the map; text-linked articles remain in the Connected rail.

## Run locally

From the `history_ML` directory, install and start the API:

    pip install -r localhistory/requirements.txt -r localhistory/api_requirements.txt
    python -m uvicorn localhistory.api:app --reload --port 8000

In a second terminal:

    cd localhistory/web
    npm install
    npm run dev

Open `http://localhost:5173`. To use another API address, copy `.env.example` to
`.env.local` and change `VITE_API_URL`.

## Structure

- `src/App.tsx` owns search and view state.
- `src/MapView.tsx` owns MapLibre and marker clustering.
- `src/api.ts` is the complete API client.
- `../api.py` adapts the existing search output for the browser.

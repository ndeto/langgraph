# AtlasAI Frontend Scaffold

Minimal Vite + React + TypeScript migration target for the AtlasAI demo UI.

## Commands

- `npm install`
- `npm run dev`
- `npm run build`
- `npm run typecheck`
- `npm run test`

## Notes

- Proxies `/api`, `/invoke`, `/ingest`, and `/health` to `http://127.0.0.1:8000` in development.
- Prefers the planned `session/thread/message` API flow.
- Falls back to the current `/invoke` and `/ingest/pdf` routes when the new thread endpoints are not available yet.

# Offline ML Semester Projects

This workspace contains two offline-first machine learning semester projects:

- `crisislens`: disaster tweet classification, clustering, API, report, and SvelteKit UI source.
- `leafguard`: crop disease image classification, generated BMP dataset, API, report, and SvelteKit UI source.

## One Command

```bash
make run
```

Alternative:

```bash
npm run start
```

Both commands train the models, refresh report artifacts, and start both APIs and UIs:

- CrisisLens: `http://127.0.0.1:8011`
- LeafGuard: `http://127.0.0.1:8022`
- CrisisLens UI: `http://127.0.0.1:5173`
- LeafGuard UI: `http://127.0.0.1:5174`

Press `Ctrl+C` to stop all services.

## Useful Commands

```bash
make train
make test
make build-ui
make verify
make crisislens
make leafguard
```

The Python ML/API layer uses only the standard library. SvelteKit dependencies are installed inside each `ui/` folder.

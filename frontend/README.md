# Prompt Optimization Framework - Frontend

A Next.js frontend for testing and comparing different prompting techniques.

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Prerequisites

Make sure the backend API is running:
```bash
cd ..
uvicorn main:app --reload
```

The API should be accessible at `http://localhost:8000`.

## Features

- Input custom problems and expected answers
- Run benchmarks comparing different prompting techniques
- View detailed results including:
  - Best performing technique
  - Accuracy, completeness, and efficiency scores
  - Response comparison across all techniques
  - Performance metrics (latency, tokens)

## Tech Stack

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS

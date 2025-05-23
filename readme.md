RUN
uvicorn indobot:app --reload --host 0.0.0.0 --port 8000

TEST
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"jam operasionalnya?"}'
RUN
<!-- uvicorn cs:app --reload --host 0.0.0.0 --port 8082 --> Deprecated
python3 app.py

TEST
curl -X POST http://localhost:8082/chat -H "Content-Type: application/json" -d '{"message":"jam operasionalnya?"}'

DOCKER SETUP
docker build -t cs-app .
docker run -p 8082:8082 cs-app
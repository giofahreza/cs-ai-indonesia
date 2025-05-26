RUN
<!-- uvicorn cs:app --reload --host 0.0.0.0 --port 8082 --> Deprecated
python3 app.py

TEST
curl -X POST http://localhost:8082/chat -H "Content-Type: application/json" -d '{"message":"jam operasionalnya?"}'

DOCKER SETUP
git clone https://github.com/giofahreza/cs-ai-indonesia.git
cp faqs_example.json faqs.json
docker build -t cs-app-image .
docker run -p 8082:8082 cs-app-image
docker container run --name cs-app-container --rm -itd -e PORT=8082 -p 8082:8082 cs-app-image
docker logs -f my-container

ADD FAQS (after create docker container)
docker exec -it cs-ai-indonesia bash
nano faqs.json
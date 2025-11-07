.PHONY: build up down logs restart clean

build:
	docker compose -f docker/docker-compose.yml build

up:
	docker compose -f docker/docker-compose.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f

restart: down up

clean:
	docker compose -f docker/docker-compose.yml down -v
	docker system prune -f

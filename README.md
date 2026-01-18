# ARM

## Running Airflow

### 1. Build and Run with Docker

```bash
docker build -t airflow:1.0 .
docker-compose up -d
```

### 2. Access Information

- URL: http://localhost:8080
- Credentials: admin / admin

> Password configuration can be found in `airflow/config/simple_auth_manager_passwords.json`

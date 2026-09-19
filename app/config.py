from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://darukaa:darukaa@localhost:5433/darukaa"
    embedding_model: str = "all-MiniLM-L6-v2"
    google_api_key: str = ""
    top_k_evidence: int = 4


settings = Settings()

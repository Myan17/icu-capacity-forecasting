from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    APP_ENV: str = "dev"
    ALERT_OCCUPANCY_YELLOW: float = 0.75
    ALERT_OCCUPANCY_RED: float = 0.90
    FORECAST_HOURS: int = 24

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
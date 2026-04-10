from violations.ml.config import CONFIG
# Add this at the end of config.py
if __name__ == "__main__":
    print("--- Config Test Success ---")
    print(f"Base Directory: {CONFIG.BASE_DIR}")
    print(f"Model Path: {CONFIG.model_path}")
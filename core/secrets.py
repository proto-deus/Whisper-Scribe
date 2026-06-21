SERVICE_NAME = "whisper-scribe"
HF_TOKEN_KEY = "hf_token"


def get_hf_token() -> str | None:
    try:
        import keyring
        token = keyring.get_password(SERVICE_NAME, HF_TOKEN_KEY)
        if token:
            return token
    except Exception:
        pass
    return None


def set_hf_token(token: str) -> None:
    try:
        import keyring
        if token:
            keyring.set_password(SERVICE_NAME, HF_TOKEN_KEY, token)
        else:
            try:
                keyring.delete_password(SERVICE_NAME, HF_TOKEN_KEY)
            except Exception:
                pass
    except Exception:
        pass


def delete_hf_token() -> None:
    try:
        import keyring
        keyring.delete_password(SERVICE_NAME, HF_TOKEN_KEY)
    except Exception:
        pass

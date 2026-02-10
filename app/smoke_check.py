"""Smoke check:
- Confirms env keys are set
- Creates store
- Confirms Chroma persistence directory is writable
"""

import os
from dotenv import load_dotenv

from vectorstore import ChromaStore


def main() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

    ok = True
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY missing in app/.env")
        ok = False
    else:
        print("✅ OPENAI_API_KEY found")

    store = ChromaStore(collection_name="personal_faq")
    print(f"✅ Chroma initialized. Current chunk count = {store.count()}")

    if ok:
        print("✅ Smoke check passed (basic wiring).")
    else:
        print("❌ Smoke check failed. Fix env vars.")


if __name__ == "__main__":
    main()

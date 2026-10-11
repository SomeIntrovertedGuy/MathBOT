"""MathBOT — demo: розв'язує математичні задачі з поясненням.
Стек: Streamlit + OpenAI API + st.secrets + історія в JSON.
"""
import base64
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import AuthenticationError, NotFoundError, OpenAI, RateLimitError

HISTORY_FILE = Path("history.json")
ERROR_MSG = "Не вдалося прочитати задачу — спробуй чіткіше фото."

SYSTEM_PROMPT = (
    "Ти — доброзичливий репетитор з математики для школяра. "
    "Відповідай українською. Формат відповіді строго такий:\n"
    "**Відповідь:** <коротка фінальна відповідь>\n\n"
    "**Розв'язання по кроках:**\n1. ...\n2. ...\n"
    "Якщо задачу неможливо прочитати або це не математична задача — "
    "відповідай рівно одним словом: UNREADABLE."
)


# ---------- допоміжні функції ----------
def get_client() -> OpenAI | None:
    """Ключ береться з .streamlit/secrets.toml (локально) або зі Secrets (Streamlit Cloud)."""
    try:
        # OPENAI_BASE_URL необов'язковий: для справжнього OpenAI його не вказують,
        # для Gemini/Groq/GitHub Models — вказують адресу їхнього OpenAI-сумісного API.
        return OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            base_url=st.secrets.get("OPENAI_BASE_URL") or None,
        )
    except (KeyError, FileNotFoundError):
        return None


def load_data() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_data(history: list[dict]) -> None:
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def solve(client: OpenAI, text: str | None, image_bytes: bytes | None, mime: str) -> str:
    content = []
    if text:
        content.append({"type": "text", "text": f"Задача: {text}"})
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode()
        content.append({"type": "text", "text": "Розв'яжи задачу з фото."})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------- інтерфейс ----------
st.set_page_config(page_title="MathBOT", page_icon="🧮")
st.title("🧮 MathBOT")
st.caption("Завантаж фото задачі або напиши її — отримаєш відповідь і пояснення по кроках.")

client = get_client()
if client is None:
    st.error("Не знайдено OPENAI_API_KEY. Додай його в `.streamlit/secrets.toml` (див. README).")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = load_data()

tab_photo, tab_text = st.tabs(["📷 Фото", "✍️ Текст"])
with tab_photo:
    photo = st.file_uploader("Фото задачі (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if photo:
        st.image(photo, caption="Твоя задача")
with tab_text:
    # План Б з паспорта: якщо AI погано читає рукопис — вставити текст вручну
    typed = st.text_area("Або напиши задачу тут", placeholder="Наприклад: розв'яжи 2x + 5 = 17")

if st.button("Відповісти", type="primary"):
    if not photo and not typed.strip():
        st.warning("Спочатку завантаж фото або напиши задачу.")
    else:
        try:
            with st.spinner("AI читає задачу..."):
                answer = solve(
                    client,
                    typed.strip() or None,
                    photo.getvalue() if photo else None,
                    photo.type if photo else "image/jpeg",
                )
            if answer.strip().upper().startswith("UNREADABLE"):
                st.error(ERROR_MSG)
            else:
                st.markdown(answer)
                st.session_state.history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%d.%m.%Y %H:%M"),
                        "task": typed.strip() or f"[фото: {photo.name}]",
                        "answer": answer,
                    },
                )
                save_data(st.session_state.history)
        except AuthenticationError:
            st.error("Невірний OPENAI_API_KEY. Перевір ключ у `.streamlit/secrets.toml`.")
        except RateLimitError as e:
            st.error("Ліміт або кошти на рахунку OpenAI вичерпано (insufficient_quota). "
                     "Перевір Billing на platform.openai.com.")
            st.caption(str(e))
        except NotFoundError as e:
            st.error("Модель не знайдена для цього провайдера. Перевір `OPENAI_MODEL` "
                     "у secrets: для Gemini — `gemini-2.5-flash`, для OpenAI — `gpt-4o-mini`.")
            st.caption(str(e))
        except Exception as e:
            st.error(ERROR_MSG)
            with st.expander("Технічні деталі"):
                st.code(f"{type(e).__name__}: {e}")

# Історія (Nice to Have з паспорта, допоміжна технологія — JSON)
with st.sidebar:
    st.header("🕘 Історія")
    if not st.session_state.history:
        st.write("Поки порожньо.")
    for item in st.session_state.history[:10]:
        with st.expander(f"{item['time']} — {item['task'][:30]}"):
            st.markdown(item["answer"])
    if st.session_state.history and st.button("Очистити історію"):
        st.session_state.history = []
        save_data([])
        st.rerun()
